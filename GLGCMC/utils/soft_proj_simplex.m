function z = soft_proj_simplex(x, beta)
    x = x - max(x);  % 数值稳定
    z = exp(beta * x) / sum(exp(beta * x));
end