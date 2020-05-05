%Start Stream
fclose all;
clear
close all

%Read Node and Edge file1
loadbasecase;
%% assign weights for lines according to nodes priority...
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
G.Edges.Weight= W_line; % take node priority as weights of lines
% G.Edges.Weight= z;%1./z; % take impedance as weights of lines;
numnodes = length(adj); %number of nodes
d=distances(G); % Diameter
maxd_nodes=max(d)' % the maximum distance for each node
Dia = max(max(d)); % the maximum distance
deg = degree(G); %

% %% Impedance weights, centrality measures
% % betweenness centrality
% figure(1)
% wbc= centrality(G,'betweenness','Cost',G.Edges.Weight); %the number of shortest path passing through node i
% wbc= wbc./sum(sum(d)); % divided by the sum of shortest paths
% p=plot(G)%, 'XDATA', nodelong, 'YDATA', nodelat)
% p.NodeCData= wbc;
% colormap jet
% colorbar
% title('Betweeness Centrality-Impedance Weight','FontSize',15)
% % [s_wbc, index1]=sort(wbc, 'descend');
% % rank_wbc=nodes(index1);
% 
% % closeness
% figure(2)
% wcc= centrality(G,'closeness','Cost',G.Edges.Weight);
% p=plot(G)%, 'XDATA', nodelong, 'YDATA', nodelat)
% p.NodeCData= wcc;
% colormap jet
% colorbar
% title('Closeness Centrality-Impedance Weight','FontSize',15)
% % [s_wcc, index2]=sort(wcc, 'descend');
% % rank_wcc=nodes(index2);
% 
% % eigenvector centrality
%  G.Edges.Importance=1./G.Edges.Weight
% %  [s_edge_importance,index_importance]=sort(G.Edges.Weight);
% %  rank_importance=nodes(index_importance);
% figure(3)
% wec= centrality(G,'eigenvector','Importance',G.Edges.Importance);
% p=plot(G);
% p.NodeCData= wec;
% colormap jet
% colorbar
% title('Eigenvector Centrality-Weighted','FontSize',15)
% % [s_wec, index3]=sort(wec, 'descend');
% % rank_wec=nodes(index3);
% 
% %% Combining above measures
% m1= (mean(wbc)/var(wbc))/((mean(wbc)/var(wbc))+(mean(wcc)/var(wcc))+(mean(wec)/var(wec)));
% m2= (mean(wcc)/var(wcc))/((mean(wbc)/var(wbc))+(mean(wcc)/var(wcc))+(mean(wec)/var(wec)));
% m3= (mean(wec)/var(wec))/((mean(wbc)/var(wbc))+(mean(wcc)/var(wcc))+(mean(wec)/var(wec)));
% 
% figure(4)
% complex_w=m1.*wbc+m2.*wcc+m3.*wec;
% % [s_complex_w,index_complex]=sort(complex_w, 'descend');
% p=plot(G);
% p.NodeCData= complex_w;
% colormap jet
% colorbar
% title('Node Comprehensive Centrality-Weighted','FontSize',15)
% % rank_complex=nodes(index_complex);


%% Load priority weights, centrality measures
% betweenness centrality
figure(1)
wbc= centrality(G,'betweenness','Cost',G.Edges.Weight); %the number of shortest path passing through node i
wbc= wbc./sum(sum(d)); % divided by the sum of shortest paths
p=plot(G)%, 'XDATA', nodelong, 'YDATA', nodelat)
p.NodeCData= wbc;
colormap jet
colorbar
title('Betweeness Centrality-Load Priority Weight','FontSize',15)
% [s_wbc, index1]=sort(wbc, 'descend');
% rank_wbc=nodes(index1)

% closeness
figure(2)
wcc= centrality(G,'closeness','Cost',G.Edges.Weight);
p=plot(G)%, 'XDATA', nodelong, 'YDATA', nodelat)
p.NodeCData= wcc;
colormap jet
colorbar
title('Closeness Centrality-Load Priority Weight','FontSize',15)
% [s_wcc, index2]=sort(wcc, 'descend');
% rank_wcc=nodes(index2);

% eigenvector centrality
 G.Edges.Importance=1./G.Edges.Weight;
 [s_edge_importance,index_importance]=sort(G.Edges.Weight);
 rank_importance=nodes(index_importance);

wec= centrality(G,'eigenvector','Importance',G.Edges.Importance);
p=plot(G);
p.NodeCData= wec;
colormap jet
colorbar
title('Eigenvector Centrality-Load Priority Weight','FontSize',15)
% [s_wec, index3]=sort(wec, 'descend');
% rank_wec=nodes(index3);

%% Combining above measures
m1= (mean(wbc)/var(wbc))/((mean(wbc)/var(wbc))+(mean(wcc)/var(wcc))+(mean(wec)/var(wec)));
m2= (mean(wcc)/var(wcc))/((mean(wbc)/var(wbc))+(mean(wcc)/var(wcc))+(mean(wec)/var(wec)));
m3= (mean(wec)/var(wec))/((mean(wbc)/var(wbc))+(mean(wcc)/var(wcc))+(mean(wec)/var(wec)));

complex_w=m1.*wbc+m2.*wcc+m3.*wec;
% [s_complex_w,index_complex]=sort(complex_w, 'descend');
% rank_complex=nodes(index_complex);
p=plot(G);
p.NodeCData= complex_w;
colormap jet
colorbar
title('Node Comprehensive Centrality-Load Priority Weight','FontSize',15)


