wavenumbers = [2.0, 4.0, 8.0, 16.0]

fig = Figure(resolution=(1600, 400), fontsize=24)
n_boot = 10000
cil = 0.99
x = (1:256)
min_x = 2^0
max_x = 2^8
min_y = 1e-8
max_y = 1e-1
basepath = "/home/kdeck/diffusion-bridge-downscaling/CliMAgen/examples/context2dturbulence/stats/"
# low res. data
spectra = readdlm(joinpath(lr_stats_basedir, "train/corr_spectrum_0.0.csv"),',')
spectra_lr = [spectra[i,:] for i in 1:802]
real_l_lr, real_u_lr = get_spectra_bci(spectra_lr, n_boot, cil)

for i in 1:4
    wn = wavenumbers[i]
    spectra = readdlm(joinpath(hr_stats_basedir, "train/corr_spectrum_$(wn).csv"),',')
    spectra_real = [spectra[i,:] for i in 1:802]
    spectra = readdlm(joinpath(hr_stats_basedir, "downscale_gen/corr_spectrum_$(wn).csv"),',')
    spectra_false = [spectra[i,:] for i in 1:800]
    real_l, real_u = get_spectra_bci(spectra_real, n_boot, cil)
    fake_l, fake_u = get_spectra_bci(spectra_false, n_boot, cil)
    ax = Axis(fig[1,i], xlabel="Wavenumber", ylabel="Average power spectrum", title=L"k_x = k_y = %$(Int(wavenumbers[i]))", xscale = log2, yscale = log10)
    band!(x, real_l, real_u, color=(:orange, 0.3), label="real high res.")
    lines!(x, real_l, color=(:orange, 0.5), strokewidth = 1.5)
    lines!(x, real_u, color=(:orange, 0.5), strokewidth = 1.5)
    band!(x, fake_l, fake_u, color=(:purple, 0.3), label="generated high res.")
    lines!(x, fake_l, color=(:purple, 0.5), strokewidth = 1.5)
    lines!(x, fake_u, color=(:purple, 0.5), strokewidth = 1.5)
    band!(x, real_l_lr, real_u_lr, color=(:green, 0.1), label="real low res.")
    lines!(x, real_l_lr, color=(:green, 0.2), strokewidth = 1.5)
    lines!(x, real_u_lr, color=(:green, 0.2), strokewidth = 1.5)
    xlims!(ax, min_x, max_x)
    ylims!(ax, min_y, max_y)
end
axislegend(; position= :rt, labelsize= 16)

save("fig:spectra_corr.png", fig, px_per_unit = 2)
