function [indicator] = BuildIndicator_my(V, N, ratio)
if (ratio < 0 || ratio > 1)
    fprintf('error: missingg rate out of the range');
end
indicator = ones(N, V);
for i=1:V
    rand_index = randperm(N);
    rand_index = rand_index(1:int32(ratio*N));
    binary_index = ones(N,1);
    binary_index(rand_index) = 0;
    indicator(:,i)= binary_index;
    clear binary_index;
end

end
