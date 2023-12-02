# fig
fig = Figure(resolution=(2000, 400), fontsize=24)
n_boot = 10000
n_grid = 200
cil = 0.99

# mean
if ch == 1
    min_x, max_x = -10, -1
else
    min_x, max_x = -20e-6, 20e-6
    xticknames = [L"-2",L"-1", L"0",L"1",L"2"]
    xticks = ([-2e-5, -1e-5, 0, 1e-5, 2e-5,], xticknames)
end
real_l_lr, real_u_lr = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== 0.0) .&& (stats.channel .== ch), :].mean, min_x, max_x, n_grid, n_boot, cil)

wn = 2.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn) .&& (stats.channel .== ch), :].mean, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn) .&& (stats.channel .== ch), :].mean, min_x, max_x, n_grid, n_boot, cil)
if ch == 1
    ax = Axis(fig[1,1], xlabel="Spatial mean supersaturation", ylabel="Probability density", title=L"k_x = k_y = 2")
else
    ax = Axis(fig[1,1], xlabel="Spatial mean vorticity (x 10⁻⁵)", ylabel="Probability density", title=L"k_x = k_y = 2", xticks = xticks)
end
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), label="real high res.")
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), label="generated high res.")
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), real_l_lr, real_u_lr, color=(:green, 0.1), label="real low res.")
lines!(LinRange(min_x, max_x, n_grid), real_l_lr, color=(:green, 0.2), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u_lr, color=(:green, 0.2), strokewidth = 1.5)
xlims!(ax, min_x, max_x)

wn = 4.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn) .&& (stats.channel .== ch), :].mean, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn) .&& (stats.channel .== ch), :].mean, min_x, max_x, n_grid, n_boot, cil)
if ch == 1
    ax = Axis(fig[1,2], xlabel="Spatial mean supersaturation", title=L"k_x = k_y = 4", yticklabelsvisible = false, titlefont = :regular)
else
    ax = Axis(fig[1,2], xlabel="Spatial mean vorticity (x 10⁻⁵)", title=L"k_x = k_y = 4", yticklabelsvisible = false, titlefont = :regular, xticks = xticks)
end
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), label="real high res.")
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), label="generated high res.")
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), real_l_lr, real_u_lr, color=(:green, 0.1), label="real low res.")
lines!(LinRange(min_x, max_x, n_grid), real_l_lr, color=(:green, 0.2), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u_lr, color=(:green, 0.2), strokewidth = 1.5)
xlims!(ax, min_x, max_x)

wn = 8.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn) .&& (stats.channel .== ch), :].mean, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn) .&& (stats.channel .== ch), :].mean, min_x, max_x, n_grid, n_boot, cil)
if ch == 1
    ax = Axis(fig[1,3], xlabel="Spatial mean supersaturation", title=L"k_x = k_y = 8", yticklabelsvisible = false, titlefont = :regular)
else
    ax = Axis(fig[1,3], xlabel="Spatial mean vorticity (x 10⁻⁵)", title=L"k_x = k_y = 8", yticklabelsvisible = false, titlefont = :regular, xticks = xticks)
end
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), label="real high res.")
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), label="generated high res.")
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), real_l_lr, real_u_lr, color=(:green, 0.1), label="real low res.")
lines!(LinRange(min_x, max_x, n_grid), real_l_lr, color=(:green, 0.2), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u_lr, color=(:green, 0.2), strokewidth = 1.5)
xlims!(ax, min_x, max_x)

wn = 16.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn) .&& (stats.channel .== ch), :].mean, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn) .&& (stats.channel .== ch), :].mean, min_x, max_x, n_grid, n_boot, cil)
if ch == 1
    ax = Axis(fig[1,4], xlabel="Spatial mean supersaturation", title=L"k_x = k_y = 16", yticklabelsvisible = false, titlefont = :regular)
else
    ax = Axis(fig[1,4], xlabel="Spatial mean vorticity (x 10⁻⁵)", title=L"k_x = k_y = 16", yticklabelsvisible = false, titlefont = :regular, xticks = xticks)
end
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), label="real high res.")
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), label="generated high res.")
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), real_l_lr, real_u_lr, color=(:green, 0.1), label="real low res.")
lines!(LinRange(min_x, max_x, n_grid), real_l_lr, color=(:green, 0.2), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u_lr, color=(:green, 0.2), strokewidth = 1.5)
xlims!(ax, min_x, max_x)
axislegend(; position= :lt, labelsize=16)

save("fig:mean_ch$(ch).png", fig, px_per_unit = 2)