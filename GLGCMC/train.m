function [Z,obj,history] = PRTL(X, cls_num, gt, opts)

    %% Parameter settings
    N = size(X{1}, 2); % the number of samples
    K = length(X);     % the number of views
    
    epsilon = 1e-5;
    max_theta = 1e10;
    max_mu = 1e10;
   
    if  isfield(opts, 'maxIter');       maxIter = opts.maxIter;         end
    if  isfield(opts, 'yita');          yita = opts.yita;               end
    if  isfield(opts, 'mu_rate');       mu_rate = opts.mu_rate;         end
    if  isfield(opts, 'theta_rate');    theta_rate = opts.theta_rate;   end
    if  isfield(opts, 'nb_num');        nb_num = opts.nb_num;           end
    if  isfield(opts, 'd');             d = opts.d;                     end
    if  isfield(opts, 'lamda1');        lamda1 = opts.lamda1;           end
    if  isfield(opts, 'lamda2');        lamda2 = opts.lamda2;           end
    %if  isfield(opts, 'alpha');         alpha = opts.alpha;            end
    if  isfield(opts, 'theta');         theta = opts.theta;             end
    if  isfield(opts, 'mu');            mu = opts.mu;                   end
    if  isfield(opts, 'm');             m = opts.m;                     end
    %% Initialize...
  
    iter = 1;
    Isconverg = 0;
    
    for k=1:K
        Z{k} = zeros(m,N);
        Q{k} = zeros(m,N);
        L{k} = zeros(N,N);
        P{k} = zeros(size(X{k},1),d);
        A{k} = zeros(size(X{k},1),m);
        B{k}=zeros(m,N);

        M{k} = zeros(m,N);
        C{k} = zeros(m,N);
    end
    B{K+1}=zeros(m,N);
    M{K+1}=zeros(m,N);
    Z_unify=zeros(m,N);
    A_unify=zeros(d,m);
    alpha = ones(1,K)/K;
    %B_tensor=cat(3, B{:,:});
    %M_tensor=cat(3, M{:,:});
    Z_tensor = cat(3, Z{:,:});
    Z_combined_tensor = cat(3, Z_tensor, reshape(Z_unify, [m, N, 1]));

    directions = [2];
    directions_number = length(directions);

    %-----------initialize G,Gamma-----------------%


    for i = 1:directions_number
        index        = directions(i);
        G{index}     = porder_diff(Z_combined_tensor,index);
        Gamma{index} = zeros(size(Z_combined_tensor));
    end
    t=tic;
    % ------------------- Update L^k -------------------------------
%         for k=1:K
%             
%             S=Z{k}'*Z{k};
%             %Weight{k} = my_constructW_PKN((abs(S)+abs(S'))./2, nb_num);
%             %Weight{k}=eye(N,N)-(abs(S)+abs(S'))./2;
%            % Diag_tmp = diag(sum(Weight{k}));
%             L{k} = eye(N,N)-(abs(S)+abs(S'))./2;
% 
%         end
%         clear S
%         time.L(iter)=toc(t);

    while(Isconverg == 0)
       
        

        t=tic;
        %% ------------------- Update Z^k -------------------------------

        for k=1:K
            tmp = (2*A{k}')*X{k}+mu*Q{k}+theta*B{k}-C{k}-M{k};
            Z_tmp = (2+(mu+theta))\tmp;
            
            for j = 1:N
                 Z{k}(:,j) = EProjSimplex_new(Z_tmp(:,j), 1);
            end

%             for ic = 1:size(Z{k},1)
%                 ind = 1:size(Z{k},2);
%                 Z{k}(ic,ind) = EProjSimplex_new(Z_tmp(ic,ind));
% 
%             end
        end
        %Z_tensor=cat(3, Z{:,:});
        clear tmp Z_tmp
        time.Zk(iter)=toc(t);

        t=tic;
        %% ------------------- Update A^k -------------------------------
        for k=1:K
            A_tmp = X{k}*Z{k}';
            [U, S, V] = svd(full(A_tmp),'econ');
            A{k}=U * V';
            %clear A_tmp
        end
         time.Ak(iter)=toc(t);

        
        t=tic;
        %% ------------------- Update P^k -------------------------------
        for k=1:K
            al2 = alpha(k)^2;
            P_tmp=al2*X{k}*Z_unify'*A_unify';
            [U, S, V] = svd(P_tmp,'econ');
            P{k}=U * V';

        end
        clear P_tmp
        time.Pk(iter)=toc(t);

        t=tic;
        %% ------------------- Update A_unify -------------------------------
        A_unify_tmp=0;
        for k=1:K
            al2 = alpha(k)^2;
            A_unify_tmp=A_unify_tmp+al2*P{k}'*X{k}*Z_unify';
        end
        [U, S, V] = svd(A_unify_tmp,'econ');
        A_unify=U * V';
        clear A_unify_tmp
        time.A(iter)=toc(t);

        t=tic;
        %% ------------------- Update Z_unify -------------------------------
        % Method 1
        H = 2*sum(alpha.^2)*eye(m,m);
        H = (H+H')/2;
        opts = optimoptions('quadprog','Display','off');  
        parfor ji=1:N
            ff=0;       
            for k=1:K 
                al2 = alpha(k)^2;
                ff = ff - 2*al2*X{k}(:,ji)'*P{k}*A_unify;
                %ff = ff - 2 * al2 * A_unify' * P{k}' * X{k}(:, ji);  
            end
            Z_unify(:,ji) = quadprog(H,ff',[],[],ones(1,m),1,zeros(m,1),[],[],opts);
            
        end
         % Method 2
%          H_scalar = 2 * sum(alpha.^2);  % H = H_scalar * I  
%          Z_unify = zeros(m, N);  
%     
%          for ji = 1:N  
%              % Build the linear term
%              ff = zeros(m, 1);
%              for k = 1:K
%                  al2 = alpha(k)^2;
%                  ff = ff - 2 * al2 * A_unify' * P{k}' * X{k}(:, ji);
%              end
% 
%              % Analytic solution: project onto the simplex
%              % Unconstrained optimum: z* = -ff/H_scalar
%              z_unconstrained = -ff / H_scalar;
% 
%              % Project onto the simplex {z: z>=0, sum(z)=1}
%              Z_unify(:, ji) = EProjSimplex_new(z_unconstrained, 1);
%          end  
 
%         % Method 3
%         tmp = zeros(m, N);
%         for k=1:K
%             al2=alpha(k)^2;
%             tmp=tmp+(2*al2*A_unify'*P{k}'*X{k})/(2*al2+theta);
%             tmp=tmp+(theta*B{K+1}-M{K+1})/(al2+theta);
%         end
%         for ic = 1:size(Z_unify,1)
%             ind = 1:size(Z{k},2);
%                  Z_unify(ic,ind) = EProjSimplex_new(tmp(ic,ind));
%             
%             
%         end
%         % Method 3
%         tmp=zeros(m,N);
%         for k=1:K  
%              tmp=tmp+A_unify\(P{k}'*X{k});
%         end
%         for ic = 1:size(Z_unify,1)
%             ind = 1:size(Z{k},2);
%             Z_unify(ic,ind) = EProjSimplex_new(tmp(ic,ind));
% 
% 
%         end
        
        
        time.Z(iter)=toc(t);

        

        t=tic;
        %% ------------------- Update Q^k -------------------------------

        for k=1:K
            %Q{k} = (mu*Z{k}+C{k})/(lamda1*(L{k}+L{k}')+mu*eye(N,N));
            % Q{k}(isnan(Q{k})) = 0; Q{k}(isinf(Q{k})) = 0;
            cons=2*lamda1+mu;
            tem0=mu*Z{k}+C{k};
            tem1=Z{k}'/cons;
            tem2=-1/(2*lamda1)+(Z{k}*Z{k}')/cons;
            tem2 = (tem2+tem2')/2;          % Symmetrize to avoid numerical asymmetry
            tem2 = tem2 + 1e-8*eye(m);      % Regularize to avoid ill-conditioning or singularity

            tem3=Z{k}/cons;
            Q{k}=tem0/cons-(tem0*tem1)*(tem2\tem3);
            
            
        end
        time.QK(iter)=toc(t);


        t=tic;
        %% ------------------- Update G ---------------------------------
        B_tensor=cat(3, B{:,:});
        for i = 1:directions_number
            index = directions(i);
            D=porder_diff(B_tensor,index)+Gamma{index}/theta;
            [gv,~] = wshrinkObj2(D,lamda2/(directions_number*theta),[m, N, K+1], 0, 3,yita);
            G_tensor = reshape(gv, [m, N, K+1]);
            G{index}=G_tensor;

        end
        time.G(iter)=toc(t);

        t=tic;
        %% ------------------- Update B ---------------------------------
        Z_tensor=cat(3, Z{:,:});
        Z_combined_tensor = cat(3, Z_tensor, reshape(Z_unify, [m, N, 1]));
        %% FFT setting
        T = zeros(size(Z_combined_tensor));
        for i = 1:directions_number
            Eny = diff_element(size(Z_combined_tensor),directions(i));
            T   = T + Eny;
        end

        HB = zeros(size(Z_combined_tensor));
        for i = 1:directions_number
            index = directions(i);
            HB = HB + porder_diff_T(theta*G{index}-Gamma{index},index);
        end

        M_tensor=cat(3, M{:,:});
        B_tensor = real( ifftn( fftn( theta*Z_combined_tensor+M_tensor+HB)./(theta*(1+T)) ) );

        for k=1:K+1
            B{k} = B_tensor(:,:,k);
        end
         time.B(iter)=toc(t);

         t=tic;
        %% ------------------- Update alpha ---------------------------------
        alpha_list = zeros(K,1);
        for k = 1:K
            alpha_list(k)= norm(P{k}'*X{k} - A_unify * Z_unify,'fro')^2;
        end
        Mfra = alpha_list.^-1;
        R = 1/sum(Mfra);
        alpha = R*Mfra;
        time.alpha(iter)=toc(t);
        
        t=tic;
        %% ------------------- Update auxiliary variables ---------------

        for k=1:K
            C{k} = C{k} + mu*(Z{k}-Q{k});
            M{k} = M{k} + theta*(Z{k}-B{k});
        end
        M{k+1} = M{k+1} + theta*(Z_unify-B{k+1});
        %M_tensor=cat(3, M{:,:});

        for i = 1:directions_number
            index = directions(i);
            Gamma{index} = Gamma{index}+theta*(porder_diff(B_tensor,index)-G{index});
        end
        time.auxiliary(iter)=toc(t);

        %% ------------------- Update penalty params --------------------
        theta = min(theta*theta_rate, max_theta);
        mu = min(mu*mu_rate, max_mu);
        %% ------------------- Converge check ---------------------------
        Z_combined_cell = [Z, Z_unify];
        Isconverg = 1;% 1: stop, 0: continue
    

        for k=1:K
            if (norm(B{k}-Z_combined_cell{k}, inf)>epsilon) || iter<10
                history.norm_B_Z(iter) = norm(B{k}-Z_combined_cell{k}, inf);
                
                Isconverg = 0;
            end
            
            if (norm(Z{k}-Q{k}, inf)>epsilon) || iter <10
                history.norm_Z_Q(iter) = norm(Z{k}-Q{k}, inf);
                Isconverg = 0;
            end

        end
     

        for i=1:directions_number
            index = directions(i);
            % Compute the partial derivative of tensor B_tensor along the index direction
            tmp = porder_diff(B_tensor,index);
            % Flatten the result into a column vector
            tmp = tmp(:);
            tmp1= G{index};
            tmp1 = tmp1(:);
            if (norm(tmp-tmp1,inf)>5e-3)|| iter<10
                
                history.norm_G_B(iter) = norm(tmp-tmp1,inf);
                Isconverg = 0;
            end
        end

        
        if (iter>maxIter)
            Isconverg  = 1;
        end
        %% each term obj
%         term1(iter)=0;term2(iter)=0;term3(iter)=0;term4(iter)=0;term5(iter)=0;
%         for k = 1:K
%             term1(iter) = term1(iter)+norm(X{k}-A{k}*Z{k},'fro')^2;
%             term2(iter) = term2(iter)+trace(Z{k}*L{k}*Z{k}');
%             term3(iter) = term3(iter)+norm(P{k}'*X{k}-A_unify*Z_unify,'fro')^2;
%             term4(iter) = term4(iter)+norm(Z{k}-B{k},'fro')^2;
%             term5(iter) = term5(iter)+norm(Z{k}-Q{k},'fro')^2;
%         end
%         %term4(iter) = term4(iter)+norm(Z_combined_cell{K+1}-B{K+1},'fro')^2;
%         obj(iter) = max([term1(iter),term2(iter),term3(iter),term4(iter),term5(iter)]);
        obj=0;
        if mod(iter, 10) == 0
            fprintf('iter: %d  ', iter);
        end
        iter = iter + 1;

G;
    end

    %% ---------------- Clustering --------------------------------------
  

                      





   

    
    
    
