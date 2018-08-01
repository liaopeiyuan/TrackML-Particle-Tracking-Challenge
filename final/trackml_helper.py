"""TrackML dataset loading"""

__authors__ = ['Moritz Kiehn', 'Sabrina Amrouche']

import glob
import os
import os.path as op
import re
import zipfile

import pandas

CELLS_DTYPES = dict([
    ('hit_id', 'i4'),
    ('ch0', 'i4'),
    ('ch1', 'i4'),
    ('value', 'f4'),
])
HITS_DTYPES = dict([
    ('hit_id', 'i4'),
    ('x', 'f4'),
    ('y', 'f4'),
    ('z','f4'),
    ('volume_id', 'i4'),
    ('layer_id', 'i4'),
    ('module_id', 'i4'),
])
PARTICLES_DTYPES = dict([
    ('particle_id', 'i8'),
    ('vx', 'f4'),
    ('vy', 'f4'),
    ('vz', 'f4'),
    ('px', 'f4'),
    ('py', 'f4'),
    ('pz', 'f4'),
    ('q', 'i4'),
    ('nhits', 'i4'),
])
TRUTH_DTYPES = dict([
    ('hit_id', 'i4'),
    ('particle_id', 'i8'),
    ('tx', 'f4'),
    ('ty', 'f4'),
    ('tz', 'f4'),
    ('tpx', 'f4'),
    ('tpy', 'f4'),
    ('tpz', 'f4'),
    ('weight', 'f4'),
])
DTYPES = {
    'cells': CELLS_DTYPES,
    'hits': HITS_DTYPES,
    'particles': PARTICLES_DTYPES,
    'truth': TRUTH_DTYPES,
}
DEFAULT_PARTS = ['hits', 'cells', 'particles', 'truth']

def _load_event_data(prefix, name):
    """Load per-event data for one single type, e.g. hits, or particles.
    """
    expr = '{!s}-{}.csv*'.format(prefix, name)
    files = glob.glob(expr)
    dtype = DTYPES[name]
    if len(files) == 1:
        return pandas.read_csv(files[0], header=0, index_col=False, dtype=dtype)
    elif len(files) == 0:
        raise Exception('No file matches \'{}\''.format(expr))
    else:
        raise Exception('More than one file matches \'{}\''.format(expr))

def load_event_hits(prefix):
    """Load the hits information for a single event with the given prefix.
    """
    return _load_event_data(prefix, 'hits')

def load_event_cells(prefix):
    """Load the hit cells information for a single event with the given prefix.
    """
    return _load_event_data(prefix, 'cells')

def load_event_particles(prefix):
    """Load the particles information for a single event with the given prefix.
    """
    return _load_event_data(prefix, 'particles')

def load_event_truth(prefix):
    """Load only the truth information for a single event with the given prefix.
    """
    return _load_event_data(prefix, 'truth')

def load_event(prefix, parts=DEFAULT_PARTS):
    """Load data for a single event with the given prefix.
    Parameters
    ----------
    prefix : str or pathlib.Path
        The common prefix name for the event files, i.e. without `-hits.csv`).
    parts : List[{'hits', 'cells', 'particles', 'truth'}], optional
        Which parts of the event files to load.
    Returns
    -------
    tuple
        Contains a `pandas.DataFrame` for each element of `parts`. Each
        element has field names identical to the CSV column names with
        appropriate types.
    """
    return tuple(_load_event_data(prefix, name) for name in parts)

def load_dataset(path, skip=None, nevents=None, parts=DEFAULT_PARTS):
    """Provide an iterator over (all) events in a dataset.
    Parameters
    ----------
    path : str or pathlib.Path
        Path to a directory or a zip file containing event files.
    skip : int, optional
        Skip the first `skip` events.
    nevents : int, optional
        Only load a maximum of `nevents` events.
    parts : List[{'hits', 'cells', 'particles', 'truth'}], optional
        Which parts of each event files to load.
    Yields
    ------
    event_id : int
        The event identifier.
    *data
        Event data element as specified in `parts`.
    """
    # extract a sorted list of event file prefixes.
    def list_prefixes(files):
        regex = re.compile('^event\d{9}-[a-zA-Z]+.csv')
        files = filter(regex.match, files)
        prefixes = set(op.basename(_).split('-', 1)[0] for _ in files)
        prefixes = sorted(prefixes)
        if skip is not None:
            prefixes = prefixes[skip:]
        if nevents is not None:
            prefixes = prefixes[:nevents]
        return prefixes

    # TODO use yield from when we increase the python requirement
    if op.isdir(path):
        for x in _iter_dataset_dir(path, list_prefixes(os.listdir(path)), parts):
            yield x
    else:
        with zipfile.ZipFile(path, mode='r') as z:
            for x in _iter_dataset_zip(z, list_prefixes(z.namelist()), parts):
                yield x

def _extract_event_id(prefix):
    """Extract event_id from prefix, e.g. event_id=1 from `event000000001`.
    """
    return int(prefix[5:])

def _iter_dataset_dir(directory, prefixes, parts):
    """Iterate over selected events files inside a directory.
    """
    for p in prefixes:
        yield (_extract_event_id(p),) + load_event(op.join(directory, p), parts)

def _iter_dataset_zip(zipfile, prefixes, parts):
    """"Iterate over selected event files inside a zip archive.
    """
    for p in prefixes:
        files = [zipfile.open('{}-{}.csv'.format(p, _), mode='r') for _ in parts]
        dtypes = [DTYPES[_] for _ in parts]
        data = tuple(pandas.read_csv(f, header=0, index_col=False, dtype=d)
                                     for f, d in zip(files, dtypes))
        yield (_extract_event_id(p),) + data
        
"""TrackML scoring metric"""

__authors__ = ['Sabrina Amrouche', 'David Rousseau', 'Moritz Kiehn',
               'Ilija Vukotic']

import numpy
import pandas

def _analyze_tracks(truth, submission):
    """Compute the majority particle, hit counts, and weight for each track.
    Parameters
    ----------
    truth : pandas.DataFrame
        Truth information. Must have hit_id, particle_id, and weight columns.
    submission : pandas.DataFrame
        Proposed hit/track association. Must have hit_id and track_id columns.
    Returns
    -------
    pandas.DataFrame
        Contains track_id, nhits, major_particle_id, major_particle_nhits,
        major_nhits, and major_weight columns.
    """
    # true number of hits for each particle_id
    particles_nhits = truth['particle_id'].value_counts(sort=False)
    total_weight = truth['weight'].sum()
    # combined event with minimal reconstructed and truth information
    event = pandas.merge(truth[['hit_id', 'particle_id', 'weight']],
                         submission[['hit_id', 'track_id']],
                         on=['hit_id'], how='left', validate='one_to_one')
    event.drop('hit_id', axis=1, inplace=True)
    event.sort_values(by=['track_id', 'particle_id'], inplace=True)

    # ASSUMPTIONs: 0 <= track_id, 0 <= particle_id

    tracks = []
    # running sum for the reconstructed track we are currently in
    rec_track_id = -1
    rec_nhits = 0
    # running sum for the particle we are currently in (in this track_id)
    cur_particle_id = -1
    cur_nhits = 0
    cur_weight = 0
    # majority particle with most hits up to now (in this track_id)
    maj_particle_id = -1
    maj_nhits = 0
    maj_weight = 0

    for hit in event.itertuples(index=False):
        # we reached the next track so we need to finish the current one
        if (rec_track_id != -1) and (rec_track_id != hit.track_id):
            # could be that the current particle is the majority one
            if maj_nhits < cur_nhits:
                maj_particle_id = cur_particle_id
                maj_nhits = cur_nhits
                maj_weight = cur_weight
            # store values for this track
            tracks.append((rec_track_id, rec_nhits, maj_particle_id,
                particles_nhits[maj_particle_id], maj_nhits,
                maj_weight / total_weight))

        # setup running values for next track (or first)
        if rec_track_id != hit.track_id:
            rec_track_id = hit.track_id
            rec_nhits = 1
            cur_particle_id = hit.particle_id
            cur_nhits = 1
            cur_weight = hit.weight
            maj_particle_id = -1
            maj_nhits = 0
            maj_weights = 0
            continue

        # hit is part of the current reconstructed track
        rec_nhits += 1

        # reached new particle within the same reconstructed track
        if cur_particle_id != hit.particle_id:
            # check if last particle has more hits than the majority one
            # if yes, set the last particle as the new majority particle
            if maj_nhits < cur_nhits:
                maj_particle_id = cur_particle_id
                maj_nhits = cur_nhits
                maj_weight = cur_weight
            # reset runnig values for current particle
            cur_particle_id = hit.particle_id
            cur_nhits = 1
            cur_weight = hit.weight
        # hit belongs to the same particle within the same reconstructed track
        else:
            cur_nhits += 1
            cur_weight += hit.weight

    # last track is not handled inside the loop
    if maj_nhits < cur_nhits:
        maj_particle_id = cur_particle_id
        maj_nhits = cur_nhits
        maj_weight = cur_weight
    # store values for the last track
    tracks.append((rec_track_id, rec_nhits, maj_particle_id,
        particles_nhits[maj_particle_id], maj_nhits, maj_weight / total_weight))

    cols = ['track_id', 'nhits',
            'major_particle_id', 'major_particle_nhits',
            'major_nhits', 'major_weight']
    return pandas.DataFrame.from_records(tracks, columns=cols)

def score_event(truth, submission):
    """Compute the TrackML event score for a single event.
    Parameters
    ----------
    truth : pandas.DataFrame
        Truth information. Must have hit_id, particle_id, and weight columns.
    submission : pandas.DataFrame
        Proposed hit/track association. Must have hit_id and track_id columns.
    """
    tracks = _analyze_tracks(truth, submission)
    purity_rec = numpy.divide(tracks['major_nhits'], tracks['nhits'])
    purity_maj = numpy.divide(tracks['major_nhits'], tracks['major_particle_nhits'])
    good_track_idx = (0.5 < purity_rec) & (0.5 < purity_maj)
	#good_track_df = tracks[good_track_idx]
    return tracks['major_weight'][good_track_idx].sum()
	
	
def score_event_kha(truth, submission):
    """Compute the TrackML event score for a single event.
    Parameters
    ----------
    truth : pandas.DataFrame
        Truth information. Must have hit_id, particle_id, and weight columns.
    submission : pandas.DataFrame
        Proposed hit/track association. Must have hit_id and track_id columns.
    """

    tracks = _analyze_tracks(truth, submission)
    purity_rec = numpy.divide(tracks['major_nhits'], tracks['nhits'])
    purity_maj = numpy.divide(tracks['major_nhits'], tracks['major_particle_nhits'])
    good_track = (0.5 < purity_rec) & (0.5 < purity_maj)
	#sensentivity = 
    return tracks['major_weight'][good_track].sum()