%extract_parameter
%pass nodetable and edgetable
%code will create the vector with the needed value for each factor
%for the scenario
%connected generation
%load demand
%timeofyear
%threat impact factor

function RW_vector = paramextract(nodetable, edgetable, SOC,ORCA_load_level,WL_HBC,WL_PC,scenarios,numprioritynodes)

loadstotal = table2array(nodetable(:,6));
load_priority =  table2array(nodetable(:,9));
connected_generator =  table2array(nodetable(:,7));
path_redundancy= table2array(nodetable(:,19));
repair_time= table2array(nodetable(:,20));
repair_cost= table2array(edgetable(:,14));
% revise ORCA diesel power plant output 
connected_generator(8:13)=connected_generator(8:13)*ORCA_load_level*0.8;  %power factor=0.8

%% revise HBC Hydro plant's output according to Water Head Level

connected_generator(2)=0.95*1000*1*9.81*WL_HBC*0.001;% in kw
connected_generator(3)=connected_generator(2);
connected_generator(4)=connected_generator(2);
%P=μρqgh,μ=efficiency, in general in the range 0.75 to 0.95;change this value according to
% actual parameters 
%ρ=density(kg/m3),1000kg/m3 for water;
%q=water flow(m3/s);here suppose equal to 1, change this value according to
% actual parameters
%g=acceleration of gravity(9.81m/s2)
%h=falling height, head(m)

% revise Power Creek Hydro plant's output according to Water Head Level
% for scenario
connected_generator(51)=0.95*1000*1*9.81*WL_PC*0.001; % in kw
connected_generator(52)=connected_generator(51);

gen_connected = 0;
battery_capacity = 1000;
nodenames = table2array(nodetable(:,1));
% find location of the battery in the node table 

for i = 1:length(nodenames)
    if strcmp(nodenames{i,1},'N400')
        index = i;
    end
end
%% count generator number and amount
gen_num=0;
for i = 1:length(connected_generator)
    if isnan(connected_generator(i)) || connected_generator(i) == Inf
        connected_generator(i) = 0;       
    end
    if connected_generator(i)~=0
        gen_num=gen_num+1;
    end        
    gen_connected = gen_connected + connected_generator(i);
    
    if index == i
        gen_connected = gen_connected + battery_capacity*SOC;
    end
end

c1_loads = 0;
for i = 1:length(load_priority)
    if load_priority(i) == 1
        c1_loads = c1_loads + loadstotal(i);
    end
end

% season = 'winter';
% %add load factor
% if strcmp(season,'winter')
%     critical_load_demand = 0.35 * c1_loads;
% elseif strcmp(season,'summer')
     critical_load_demand =c1_loads;
% else
%    critical_load_demand = 0.55 * c1_loads;
%    disp("The input for season is wrong. Deault summer load is applied");
% end

%threat impact factor
%to be modified based on inputs/discussion
%to be hard coded for now


%% critical load not lost and critical path redundancy
total_critical_loads = numprioritynodes;
critical_loads = 0;
critical_path_redundancy=0;
for i = 1:length(load_priority)
    if load_priority(i) == 1
        critical_loads = critical_loads + 1;
        critical_path_redundancy=critical_path_redundancy+path_redundancy(i);       
    end
end
CLNL = 0;

CLNL = critical_loads/total_critical_loads;
%% critical load fraction
TotalLoads_case = 0;
for i = 1:length(load_priority)
    if isnan(load_priority(i))
        load_priority(i) = 3;
    end
    if load_priority(i)==1||load_priority(i)==2||load_priority(i)==3
        TotalLoads_case = TotalLoads_case + 1;
    end
end
CLF= critical_loads/TotalLoads_case;

 
RW_vector = [gen_connected,CLNL,critical_path_redundancy,CLF];
end
%     
% % totalload = sum(nodetable(:,6));
% totalload = table2array(varfun(@sum,nodetable(:,6)));
% connectedgen = table2array(varfun(@sum,nodetable(:,7)));