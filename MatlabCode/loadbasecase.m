%Setup file paths
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input: node-file.csv
% Output: node data
% Function: 
% Items in Path Folder:
% glmMap.py - PNNL
% Matlab Graph Codes - MIT
clear;
clc;
close;

base_dir = pwd
% '/Scenario Files/'
%%change file paths approporiately
addpath(genpath([base_dir, '/Scenario Files/']))
nodefilename = [base_dir, '/Scenario Files/Case1/node-file-case1.csv'];
edgefilename = [base_dir, '/Scenario Files/Case1/edge-file-case1.csv'];
%%check for fileread error
a = fopen(nodefilename,'r');
if a<0
    disp("File not read Error")
end
%load node table
nodetable = readtable(nodefilename);
nodenames = table2array(nodetable(:,1));
nodephases = table2array(nodetable(:,2));
nodelat = table2array(nodetable(:,3));
nodelong = table2array(nodetable(:,4));
nodevolt = table2array(nodetable(:,5));
nodeload = table2array(nodetable(:,6));
nodegen = table2array(nodetable(:,7));
nodekind = table2array(nodetable(:,8));
nodepriority = table2array(nodetable(:,9));
nodePD =  table2array(nodetable(:,17));

prioritynodes = 0;

for i = 1:length(nodepriority)
    if isnan(nodepriority(i))
        nodepriority(i) = 3;
    end
    if nodepriority(i)==1
        prioritynodes = prioritynodes + 1;
    end
end


%%check for fileread error
a = fopen(edgefilename,'r');
if a<0
    disp("File not read Error")
end
edgetable = readtable(edgefilename);
edgenames = table2array(edgetable(:,1));
edgekind = table2array(edgetable(:,2));
fromnode = table2array(edgetable(:,3));
tonode = table2array(edgetable(:,4));
edgestatus = table2array(edgetable(:,5));
dotm = strings([100,2]);
for i = 1:length(fromnode)
    dotm(i,1) = cellstr(fromnode(i,1));
    dotm(i,2) = cellstr(tonode(i,1));
end
k=1;
for i = 1:length(dotm)
    if dotm(i,1)~=''
      edgelist(k,:) = dotm(i,:);
      k = k+1;
    end
end

m = length(nodenames);
for i = 1:m
    nodenames(i) = strrep(nodenames(i),'n_','N');
end
nodes=nodenames; % get all nodes, sorted
adj=zeros(numel(nodes));   % initialize adjacency matrix

% across all edges
for i=1:size(edgelist,1)
    if edgestatus(i) ~= 0
        adj(find(nodes==edgelist(i,1)),find(nodes==edgelist(i,2)))=1;
    end
end

% % 
for i = 1:length(adj)
    for j = 1:length(adj)
        if adj(i,j)==0&&adj(j,i)==1
            adj(i,j)=1;
        end
    end
end
            
%remove underscore
%creates subscript in plots
match = '_';
nodes = erase(nodes,match);
%Create nodenames
nodes= cellstr(nodes);


%Store Base Node and Edge Tables for later modifications
BaseCaseNodeTable = nodetable;
BaseCaseEdgeTable = edgetable;