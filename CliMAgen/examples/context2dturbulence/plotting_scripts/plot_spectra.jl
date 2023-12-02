# fig
fig = Figure(resolution=(1600, 400), fontsize=24)
n_boot = 10000
cil = 0.99
x = (1:256)
min_x = 2^0
max_x = 2^8
min_y = 1e-6
max_y = ch == 1 ? 1e0 : 1e1 

# low res. data
spectra_lr = [Array(r) for r in eachrow(spectra[(spectra.isreal .== true) .&& (spectra.wavenumber .== 0.0) .&& (spectra.channel .== ch), 1:256])]
real_l_lr, real_u_lr = get_spectra_bci(spectra_lr, n_boot, cil)

wn = 2.0
spectra_real = [Array(r) for r in eachrow(spectra[(spectra.isreal .== true) .&& (spectra.wavenumber .== wn) .&& (spectra.channel .== ch), 1:256])]
spectra_false = [Array(r) for r in eachrow(spectra[(spectra.isreal .== false) .&& (spectra.wavenumber .== wn) .&& (spectra.channel .== ch), 1:256])]
real_l, real_u = get_spectra_bci(spectra_real, n_boot, cil)
fake_l, fake_u = get_spectra_bci(spectra_false, n_boot, cil)
ax = Axis(fig[1,1], xlabel="Wavenumber", ylabel="Average power spectrum", title=L"k_x = k_y = 2", xscale = log2, yscale = log10)
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

wn = 4.0
spectra_real = [Array(r) for r in eachrow(spectra[(spectra.isreal .== true) .&& (spectra.wavenumber .== wn) .&& (spectra.channel .== ch), 1:256])]
spectra_false = [Array(r) for r in eachrow(spectra[(spectra.isreal .== false) .&& (spectra.wavenumber .== wn) .&& (spectra.channel .== ch), 1:256])]
real_l, real_u = get_spectra_bci(spectra_real, n_boot, cil)
fake_l, fake_u = get_spectra_bci(spectra_false, n_boot, cil)
ax = Axis(fig[1,2], xlabel="Wavenumber", title=L"k_x = k_y = 4", xscale = log2, yscale = log10, yticklabelsvisible = false, titlefont = :regular)
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

wn = 8.0
spectra_real = [Array(r) for r in eachrow(spectra[(spectra.isreal .== true) .&& (spectra.wavenumber .== wn) .&& (spectra.channel .== ch), 1:256])]
spectra_false = [Array(r) for r in eachrow(spectra[(spectra.isreal .== false) .&& (spectra.wavenumber .== wn) .&& (spectra.channel .== ch), 1:256])]
real_l, real_u = get_spectra_bci(spectra_real, n_boot, cil)
fake_l, fake_u = get_spectra_bci(spectra_false, n_boot, cil)
ax = Axis(fig[1,3], xlabel="Wavenumber", title=L"k_x = k_y = 8", xscale = log2, yscale = log10, yticklabelsvisible = false, titlefont = :regular)
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

wn = 16.0
spectra_real = [Array(r) for r in eachrow(spectra[(spectra.isreal .== true) .&& (spectra.wavenumber .== wn) .&& (spectra.channel .== ch), 1:256])]
spectra_false = [Array(r) for r in eachrow(spectra[(spectra.isreal .== false) .&& (spectra.wavenumber .== wn) .&& (spectra.channel .== ch), 1:256])]
real_l, real_u = get_spectra_bci(spectra_real, n_boot, cil)
fake_l, fake_u = get_spectra_bci(spectra_false, n_boot, cil)
ax = Axis(fig[1,4], xlabel="Wavenumber", title=L"k_x = k_y = 16", xscale = log2, yscale = log10, yticklabelsvisible = false, titlefont = :regular)
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

axislegend(; position= :rt, labelsize= 16)

save("fig:spectra_ch$(ch).png", fig, px_per_unit = 2)
