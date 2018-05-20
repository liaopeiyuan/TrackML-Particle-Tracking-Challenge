event=[];
for i=0:9
    A=csvread(strcat('event00000000',num2str(i),'-hits.csv'),1,0);
    A=A(:,2:4);
    event=vertcat(event,A);
end

event1=event./sqrt(event(:,1).^2+event(:,2).^2+event(:,3).^2);
c=clusterdata(event1,'Linkage','ward','Savememory','on','Cutoff',1.15465);