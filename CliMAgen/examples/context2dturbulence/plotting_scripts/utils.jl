
## utils 
function get_pdf(data, min_x, max_x, n_grid)
    estimate = kde(data)
    pdf(estimate, LinRange(min_x, max_x, n_grid))
end

function get_pdf_bci(data, min_x, max_x, n_grid, n_boot, cil)
    x = LinRange(min_x, max_x, n_grid)
    bs = bootstrap(x -> get_pdf(x, min_x, max_x, n_grid), data, BasicSampling(n_boot))
    ci = confint(bs, PercentileConfInt(cil))
    lower = [c[2] for c in ci]
    upper = [c[3] for c in ci]
    return lower, upper
end

function get_spectra_bci(data, n_boot, cil)
    bs = bootstrap(mean, data, BasicSampling(n_boot))
    ci = confint(bs, PercentileConfInt(cil))
    lower = [c[2] for c in ci]
    upper = [c[3] for c in ci]
    return lower, upper
end

function get_mean_bci(data, n_boot, cil)
    bs = bootstrap(mean, data, BasicSampling(n_boot))
    ci = confint(bs, PercentileConfInt(cil))
    lower = [c[2] for c in ci]
    upper = [c[3] for c in ci]
    return lower, upper
end

function percentile_scale(x, data, lower=1.0, upper=4.0)
    expo = LinRange(lower, upper, length(x))
    quantiles = map(x -> 1-1/10^x, expo)
    x_quantiles = map(x -> quantile(data, x), quantiles)
    interp_linear = linear_interpolation(x_quantiles, quantiles)
    return interp_linear.(x)
end

heaviside(x) = x > 0 ? 1.0 : 0.0

function cdf(x,samples)
    return mean( heaviside.(x.-samples))
end