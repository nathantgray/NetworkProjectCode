    function [eigvect ] = calc_eig(M) 
        %Convert pairwise matrix (PCM) into ranking of criteria (RCM) using
        %eignevectors (reference: the analytical hierarchy process, 1990,
        % Thomas L. Saaty
        % Note: A simple/fast way to solve for the eigenvectors are:
        % 1. Raise pairwise matrix to powers that are successively squared
        % 2. Sum the rows and normalize summed rows.
        % 3. Stop when the difference between the sums in two consecutive
        %    iterations is smaller than tolerance.
        c=1;
        [m n]=size(M);
        nrM(m,:)=10000; tolmet=0; tolerance=.035;
        while tolmet<1 
            c=c+1;                                        % counter
            M=M^2;                                        % pairwise matrix^2
            sr1M = sum(M,2);                              % sum rows
            sr2M = sum(sr1M);                             % sum of sum rows
            nrM(:,c) = sr1M./sr2M;                        % normalize
            tol(c)=sum(abs(nrM(:,c) - nrM(:,c-1)));       % calc. tolerance
             if tol < tolerance                    % tolerance met?
                tolmet=1;                          % tolerance met, stop iterations
             elseif sum(sum(M))>=10e30 
                 tolmet=1;                         % tolerance unlikely, stop iterations
             end
        end
%         disp('Eigenvector of matrix');
        eigvect = nrM(:,end); % eigenvector of PCM
    end