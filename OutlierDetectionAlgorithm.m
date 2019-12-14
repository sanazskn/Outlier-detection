function OutlierIndex=OutlierDetectionAlgorithm(X,h,k,numberOutlier)
d=size(X,2);
Mdl = KDTreeSearcher(X);
[Sknn,DGraph] = knnsearch(Mdl,X,'K',k); %find K-nearest 
%reverse-nearest
Srnn={};
for i=1:size(Sknn,1)
    [Srnn{i},~]=find(Sknn(1:end,2:end)==i);
   
end
% Share-neareast
Ssnn={};
for i=1:size(Sknn,1)
    Ssnn{i}=[];
   for j=2:size(Sknn,2)
       [Srnntemp,~]=find(Sknn(1:end,2:end)==Sknn(i,j));
       Srnntemp=unique(Srnntemp);
       Ssnn{i}=[Ssnn{i}; Srnntemp];
   end    
end;

S_All={};
for i=1:size(X,1)
    S_All{i}=[Sknn(i,2:end) Srnn{i}' Ssnn{i}'];
     S_All{i}=unique(S_All{i});
     S_All{i} = setdiff(S_All{i},i);
end

DistanceMatrice=dist(X');
Kernelh=@(h,d,DistanceMatrice,i,j) (1/(2*pi)^(d/2))*exp(-DistanceMatrice(i,j).^2/(2*h^2));
P=zeros(size(X,1),1);
for i=1:size(X,1)
    j=S_All{i}; 
    summation=0;
    for p=1:numel(j)
        summation=summation+(Kernelh(h,d,DistanceMatrice,i,j(p)));
    end
    P(i)=(1/(1+numel(S_All{i})))*(1/h^d)*summation;
end
%% RDOS
for i=1:size(X,1)
     j=S_All{i};
     RDOS(i)=sum(P(j))/(numel(j)*P(i));
end
%% sort
[~,Index] = sort(RDOS,'descend');
OutlierIndex=Index(1:numberOutlier);
end