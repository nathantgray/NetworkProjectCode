%Start Stream
fclose all;
clear
close all

%Read Node and Edge file1
loadbasecase;
%assign weights for lines according to nodes priority...
%line weights equal to the bigger priority value of its nodes
for i=1:size(edgelist,1)
    if edgestatus(i) ~= 0
        m=find(nodes==edgelist(i,1));
        n=find(nodes==edgelist(i,2));
        W_line(i)=max(nodepriority(m), nodepriority(n));
    end
end
W_line = W_line'
r = table2array(edgetable(:, 6));
x = table2array(edgetable(:, 7));
z = sqrt(r.^2 + x.^2);
%[G, TopoVector] = TopologicalFactorExtraction(adj,nodes,z)

%create graph object with node names and adjacency matrix
G= graph(adj,nodes);
G.Edges.Weight=z %1./z%z(find(edgestatus));
numnodes = length(adj); %number of nodes
d=distances(G); % Diameter
maxd_nodes=max(d)' % the maximum distance for each node
Dia = max(max(d)); % the maximum distance
deg = degree(G); %

%% centrality measures

% betweenness centrality
figure(1)
wbc= centrality(G,'betweenness','Cost',G.Edges.Weight); %the number of shortest path passing through node i
wbc= wbc./sum(sum(d)); % divided by the sum of shortest paths
p=plot(G)%, 'XDATA', nodelong, 'YDATA', nodelat)
p.NodeCData= wbc;
colormap jet
colorbar
title('Betweeness Centrality-Weighted')

% closeness
figure(2)
wcc= centrality(G,'closeness','Cost',G.Edges.Weight);
p=plot(G)%, 'XDATA', nodelong, 'YDATA', nodelat)
p.NodeCData= wcc;
colormap jet
colorbar
title('Closeness Centrality-Weighted')




