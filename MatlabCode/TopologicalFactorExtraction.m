%for final project of 591, analyze centrality of a real power grid
% clear
% clc

function [G TopoVector] = TopologicalFactorExtraction(adj,nodes,W_line)
%create graph object with node names and adjacency matrix
G= graph(adj,nodes);
G.Edges.Weight=W_line;
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
p=plot(G)
p.NodeCData= wbc;
colormap jet
colorbar
title('Betweeness Centrality-Weighted')

% closeness
figure(2)
wcc= centrality(G,'closeness','Cost',G.Edges.Weight);
p=plot(G)
p.NodeCData= wcc;
colormap jet
colorbar
title('Closeness Centrality-Weighted')


% % Algebraic Connectivity
% %second smallest eigenvalue of Laplacian matrix of the network
% %Setup Laplacian
%
% L = zeros(length(adj));
% for i = 1:length(adj)
%     for j = 1:length(adj)
%         if i==j
%             L(i,i)=deg(i);
%         elseif adj(i,j)==1
%             L(i,j) = -1;
%         else L(i,j) = 0;
%         end
%     end
% end
%
% e = eig(L);
% Alg_c = e(2);
%
% % Degree Distribution
% E = numedges(G);
% K = 2*E/numnodes;
%
% % second moment
% K2 = (std(deg))^2;
% k0 = K2/K;
% %percholation threshold
% fc = 1 - 1/(k0 - 1);
% %the topological vector is ordered as per rank - [ 5 4 2 1 3 ] eg. the 1st
% %factor in topovector is the lowest rated
% TopoVector = [length(adj) K fc Alg_c CB ];
TopoVector = [Dia  ]
end