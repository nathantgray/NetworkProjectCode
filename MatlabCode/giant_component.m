function [GC,gc_nodes]=giant_component(adj)
%returns the largest connected component of the graph
%in case CB status creates disconnected nodes

comps=find_conn_comp(adj);

L=[];
for k=1:length(comps); L=[L, length(comps{k})]; end
[maxL,ind_max]=max(L);

gc_nodes=comps{ind_max};
GC=subgraph(adj,gc_nodes);