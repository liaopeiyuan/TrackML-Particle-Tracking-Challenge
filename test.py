from trackml.dataset import load_event

hits, cells, particles, truth = load_event('./portable-dataset/')

from trackml.randomize import shuffle_hits
from trackml.score import score_event

shuffled = shuffle_hits(truth, 0.05) # 5% probability to reassign a hit
score = score_event(truth, shuffled)
print(score)
