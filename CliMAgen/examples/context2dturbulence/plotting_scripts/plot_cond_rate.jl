# fig
fig = Figure(resolution=(1600, 400), fontsize=24)
n_pixels = 1600000
n_boot = 10000
n_grid = 100
cil = 0.99
FT = Float32
min_x, max_x = 0, 325
min_y, max_y = 1e-7, 1e-1
x = LinRange(min_x, max_x, n_grid)
τ = 1e-2

wn = 2.0
cr = Array(pixels[(pixels.isreal .== true) .&& (pixels.wavenumber .== wn) .&& (pixels.channel .== ch), 1:n_pixels])[:]
cr = cr[cr .> 0] / τ
q = quantile(cr, 0.9)
real_l_hr, real_u_hr = get_pdf_bci(cr, min_x, max_x, n_grid, n_boot, cil)
cr = Array(pixels[(pixels.isreal .== false) .&& (pixels.wavenumber .== wn) .&& (pixels.channel .== ch), 1:n_pixels])[:]
cr = cr[cr .> 0] / τ
fake_l_hr, fake_u_hr = get_pdf_bci(cr, min_x, max_x, n_grid, n_boot, cil)
cr = Array(pixels[(pixels.isreal .== true) .&& (pixels.wavenumber .== 0.0) .&& (pixels.channel .== ch), 1:n_pixels])[:]
cr = cr[cr .> 0] / τ
real_l_lr, real_u_lr = get_pdf_bci(cr, min_x, max_x, n_grid, n_boot, cil)
real_l_hr, real_u_hr, fake_l_hr, fake_u_hr = real_l_hr .+ eps(FT), real_u_hr .+ eps(FT), fake_l_hr .+ eps(FT), fake_u_hr .+ eps(FT)
real_l_lr, real_u_lr = real_l_lr .+ eps(FT), real_u_lr.+ eps(FT)
ax = Axis(fig[1,1], xlabel="Condensation rate", ylabel="Probability density", title=L"k_x = k_y = 2", yscale=log10, titlefont = :regular)
vlines!(ax, [q], color=(:black, 0.3))
band!(x, real_l_hr, real_u_hr, color=(:orange, 0.3), label="real high res.")
lines!(x, real_l_hr, color=(:orange, 0.5), strokewidth = 1.5)
lines!(x, real_u_hr, color=(:orange, 0.5), strokewidth = 1.5)
band!(x, fake_l_hr, fake_u_hr, color=(:purple, 0.3), label="generated high res.")
lines!(x, fake_l_hr, color=(:purple, 0.5), strokewidth = 1.5)
lines!(x, fake_u_hr, color=(:purple, 0.5), strokewidth = 1.5)
band!(x, real_l_lr, real_u_lr, color=(:green, 0.1), label="real low res.")
lines!(x, real_l_lr, color=(:green, 0.2), strokewidth = 1.5)
lines!(x, real_u_lr, color=(:green, 0.2), strokewidth = 1.5)
xlims!(ax, min_x, max_x)
ylims!(ax, min_y, max_y)

wn = 4.0
cr = Array(pixels[(pixels.isreal .== true) .&& (pixels.wavenumber .== wn) .&& (pixels.channel .== ch), 1:n_pixels])[:]
cr = cr[cr .> 0] / τ
q = quantile(cr, 0.9)
real_l_hr, real_u_hr = get_pdf_bci(cr, min_x, max_x, n_grid, n_boot, cil)
cr = Array(pixels[(pixels.isreal .== false) .&& (pixels.wavenumber .== wn) .&& (pixels.channel .== ch), 1:n_pixels])[:]
cr = cr[cr .> 0] / τ
fake_l_hr, fake_u_hr = get_pdf_bci(cr, min_x, max_x, n_grid, n_boot, cil)
cr = Array(pixels[(pixels.isreal .== true) .&& (pixels.wavenumber .== 0.0) .&& (pixels.channel .== ch), 1:n_pixels])[:]
cr = cr[cr .> 0] / τ
real_l_lr, real_u_lr = get_pdf_bci(cr, min_x, max_x, n_grid, n_boot, cil)
real_l_hr, real_u_hr, fake_l_hr, fake_u_hr = real_l_hr .+ eps(FT), real_u_hr .+ eps(FT), fake_l_hr .+ eps(FT), fake_u_hr .+ eps(FT)
real_l_lr, real_u_lr = real_l_lr .+ eps(FT), real_u_lr.+ eps(FT)
ax = Axis(fig[1,2], xlabel="Condensation rate", title=L"k_x = k_y = 4", yscale=log10, yticklabelsvisible = false, titlefont = :regular)
vlines!(ax, [q], color=(:black, 0.3))
band!(x, real_l_hr, real_u_hr, color=(:orange, 0.3), label="real high res.")
lines!(x, real_l_hr, color=(:orange, 0.5), strokewidth = 1.5)
lines!(x, real_u_hr, color=(:orange, 0.5), strokewidth = 1.5)
band!(x, fake_l_hr, fake_u_hr, color=(:purple, 0.3), label="generated high res.")
lines!(x, fake_l_hr, color=(:purple, 0.5), strokewidth = 1.5)
lines!(x, fake_u_hr, color=(:purple, 0.5), strokewidth = 1.5)
band!(x, real_l_lr, real_u_lr, color=(:green, 0.1), label="real low res.")
lines!(x, real_l_lr, color=(:green, 0.2), strokewidth = 1.5)
lines!(x, real_u_lr, color=(:green, 0.2), strokewidth = 1.5)
xlims!(ax, min_x, max_x)
ylims!(ax, min_y, max_y)

wn = 8.0
cr = Array(pixels[(pixels.isreal .== true) .&& (pixels.wavenumber .== wn) .&& (pixels.channel .== ch), 1:n_pixels])[:]
cr = cr[cr .> 0] / τ
q = quantile(cr, 0.9)
real_l_hr, real_u_hr = get_pdf_bci(cr, min_x, max_x, n_grid, n_boot, cil)
cr = Array(pixels[(pixels.isreal .== false) .&& (pixels.wavenumber .== wn) .&& (pixels.channel .== ch), 1:n_pixels])[:]
cr = cr[cr .> 0] / τ
fake_l_hr, fake_u_hr = get_pdf_bci(cr, min_x, max_x, n_grid, n_boot, cil)
cr = Array(pixels[(pixels.isreal .== true) .&& (pixels.wavenumber .== 0.0) .&& (pixels.channel .== ch), 1:n_pixels])[:]
cr = cr[cr .> 0] / τ
real_l_lr, real_u_lr = get_pdf_bci(cr, min_x, max_x, n_grid, n_boot, cil)
real_l_hr, real_u_hr, fake_l_hr, fake_u_hr = real_l_hr .+ eps(FT), real_u_hr .+ eps(FT), fake_l_hr .+ eps(FT), fake_u_hr .+ eps(FT)
real_l_lr, real_u_lr = real_l_lr .+ eps(FT), real_u_lr.+ eps(FT)
ax = Axis(fig[1,3], xlabel="Condensation rate",  title=L"k_x = k_y = 8", yscale=log10, yticklabelsvisible = false, titlefont = :regular)
vlines!(ax, [q], color=(:black, 0.3))
band!(x, real_l_hr, real_u_hr, color=(:orange, 0.3), label="real high res.")
lines!(x, real_l_hr, color=(:orange, 0.5), strokewidth = 1.5)
lines!(x, real_u_hr, color=(:orange, 0.5), strokewidth = 1.5)
band!(x, fake_l_hr, fake_u_hr, color=(:purple, 0.3), label="generated high res.")
lines!(x, fake_l_hr, color=(:purple, 0.5), strokewidth = 1.5)
lines!(x, fake_u_hr, color=(:purple, 0.5), strokewidth = 1.5)
band!(x, real_l_lr, real_u_lr, color=(:green, 0.1), label="real low res.")
lines!(x, real_l_lr, color=(:green, 0.2), strokewidth = 1.5)
lines!(x, real_u_lr, color=(:green, 0.2), strokewidth = 1.5)
xlims!(ax, min_x, max_x)
ylims!(ax, min_y, max_y)

wn = 16.0
cr = Array(pixels[(pixels.isreal .== true) .&& (pixels.wavenumber .== wn) .&& (pixels.channel .== ch), 1:n_pixels])[:]
cr = cr[cr .> 0] / τ
q = quantile(cr, 0.9)
real_l_hr, real_u_hr = get_pdf_bci(cr, min_x, max_x, n_grid, n_boot, cil)
cr = Array(pixels[(pixels.isreal .== false) .&& (pixels.wavenumber .== wn) .&& (pixels.channel .== ch), 1:n_pixels])[:]
cr = cr[cr .> 0] / τ
fake_l_hr, fake_u_hr = get_pdf_bci(cr, min_x, max_x, n_grid, n_boot, cil)
cr = Array(pixels[(pixels.isreal .== true) .&& (pixels.wavenumber .== 0.0) .&& (pixels.channel .== ch), 1:n_pixels])[:]
cr = cr[cr .> 0] / τ
real_l_lr, real_u_lr = get_pdf_bci(cr, min_x, max_x, n_grid, n_boot, cil)
real_l_hr, real_u_hr, fake_l_hr, fake_u_hr = real_l_hr .+ eps(FT), real_u_hr .+ eps(FT), fake_l_hr .+ eps(FT), fake_u_hr .+ eps(FT)
real_l_lr, real_u_lr = real_l_lr .+ eps(FT), real_u_lr.+ eps(FT)
ax = Axis(fig[1,4], xlabel="Condensation rate", title=L"k_x = k_y = 16", yscale=log10, yticklabelsvisible = false, titlefont = :regular)
vlines!(ax, [q], color=(:black, 0.3))
band!(x, real_l_hr, real_u_hr, color=(:orange, 0.3), label="real high res.")
lines!(x, real_l_hr, color=(:orange, 0.5), strokewidth = 1.5)
lines!(x, real_u_hr, color=(:orange, 0.5), strokewidth = 1.5)
band!(x, fake_l_hr, fake_u_hr, color=(:purple, 0.3), label="generated high res.")
lines!(x, fake_l_hr, color=(:purple, 0.5), strokewidth = 1.5)
lines!(x, fake_u_hr, color=(:purple, 0.5), strokewidth = 1.5)
band!(x, real_l_lr, real_u_lr, color=(:green, 0.1), label="real low res.")
lines!(x, real_l_lr, color=(:green, 0.2), strokewidth = 1.5)
lines!(x, real_u_lr, color=(:green, 0.2), strokewidth = 1.5)
xlims!(ax, min_x, max_x)
ylims!(ax, min_y, max_y)
axislegend(; position= :lb, labelsize= 16)

save("fig:cond_rate.png", fig, px_per_unit = 2)