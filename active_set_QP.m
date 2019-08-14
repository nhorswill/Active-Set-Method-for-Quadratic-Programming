function [x,W,q] = active_set_QP(G,c,A,b,x_0,W_0)
%{
    The following algorithm provides a solution to quadratic programs of
    the following form:
    
    Minimize q(x) = x'*G*x/2+c'*x
    s.t. a'*x=b
         aa'*x>=bb
    
        G is a PD symmetric matrix.  The sets containing equality
        constraints and inequality constraints (a and aa) compose matrix A,
        and likewise for b.  Constraints are linearly independent.
    
        Note that constraint inequaities take the form Ax >= b, and this
        algorithm solves minimization problems.

    The algorithm generates a set W_k of indices of linearly independent
    active constraints.  More can be found in the notes section from UNC's
    STOR 614: Linear Programming, Lecture 17.  Notes Provided courtesy of
    Shu Lu, UNC department of Operations Research.

    The solution returns a set of three elements.  The optimal variables
    are returned as x, the optimal cost function as q, and the final
    working set of active constraints is returned as W.

%}

%% initial params
q=[];
x=[x_0];
W = [W_0];
x_k = x_0;
W_k = W_0;
k = 0;
dims = size(A);
indices = 1:dims(1);
stop = 0;
    while stop==0
       W_c = setdiff(indices,W_k);
       %% set iteration parameters
       k=k+1;
       %% solve equality constrained QP
       g_k = G*x_k+c;
       dimens=size(A(W_k,:));
       if isempty(W_k)
           p = quadprog(G,g_k);
       else
           p = quadprog(G,g_k,[],[],A(W_k,:),zeros(dimens(1),1));
       end
       for i=1:length(p)
           if abs(p(i))<1e-7
               p(i)=0;
           end
       end
       %% Case 1
       if any(p ~= 0)
           change=A(W_c,:)*p;
           for k=1:length(change)
               if abs(change(k))<1e-7
                   change(k)=0;
               end
           end
           rtindex = find(change < 0);
           newindex = W_c(rtindex);
           small = (b(newindex,:)-A(newindex,:)*x_k)./(A(newindex,:)*p);
           rt = min(small);
           ak = min([1,rt]);
           x_k = x_k+ak*p;
       %% Case 1.1
           if ak==1
               continue;
           end
       %% Case 1.2
           if (ak~=1)
               j = find(small==rt);
               j=j(1);
               j1 = W_c(j);
               W_k = [W_k, j1];
           end
       end
       %% Case 2
       if all(p==0)
           lambda = linsolve(A(W_k,:)',g_k);
       %% Case 2.1
           if all(lambda >=0)
               stop=1;
           end
       %% Case 2.2
           if any(lambda<0)
               j = find(lambda<0);
               j1=j(1);
               x_k=x_k;
               W_k(j1) = [];
           end
       end
       %% save and return
       q_k = x_k'*G*x_k/2+c'*x_k;
       x = [x, x_k];
       if isempty(W_k)
           W_m=NaN;
       else
           W_m=W_k;
       end
       W = [W, W_m];
       q = [q, q_k];
    end
end