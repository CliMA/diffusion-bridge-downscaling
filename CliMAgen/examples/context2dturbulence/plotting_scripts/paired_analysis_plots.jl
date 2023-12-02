wn = 16.0
n_boot = 10000
n_grid = 200
cil = 0.99
ch = 1
if ch == 1
    min_x, max_x = -25, 5
else
    min_x, max_x = -20, 20
end
x = LinRange(min_x, max_x, n_grid)

samples_filepath = joinpath(basedir, "stats/paired_analysis/samples_$(wn).hdf5")
fid = HDF5.h5open(samples_filepath, "r")
fake_hr_samples = cat([fid["downscaled_samples_$i"][:,:,:,:] for i in 1:5]...,dims = 4)
real_hr_sample = fid["original_samples"][:,:,:,:]
real_lr_sample = fid["fake_lowres_samples"][:,:,:,:]
close(fid)

pixels = [ [128, 128],[256, 256], [384, 384]]
fig = Figure(resolution=(1600, 400), fontsize=24)
titles = ["(128,128)", "(256,256)", "(384, 384)"]

for i in 1:length(pixels)
    i == 1 ? ylabel = "Probability density" : ylabel = ""
    ax = Axis(fig[1,i], xlabel="Pixel value", ylabel=ylabel, title=L"%$(titles[i])")
    fake_hr_pixel_values = fake_hr_samples[pixels[i][1], pixels[i][2], ch,:]
    real_hr_pixel_value = real_hr_sample[pixels[i][1], pixels[i][2], ch,1]
    real_lr_pixel_value = real_lr_sample[pixels[i][1], pixels[i][2], ch,1]
    fake_l, fake_u = get_pdf_bci(fake_hr_pixel_values, min_x, max_x, n_grid, n_boot, cil)
    band!(x, fake_l, fake_u, color=(:purple, 0.3), label="gen. dist., high res.")
    lines!(x, fake_l, color=(:purple, 0.5), strokewidth = 1.5)
    lines!(x, fake_u, color=(:purple, 0.5), strokewidth = 1.5)
    ylims!(ax, 0, 0.3)
    xlims!(ax, min_x, max_x)
    lines!(real_hr_pixel_value .+ [0,0], [1e-4, 0.3], label = "true value, high res.", color=(:orange, 1.5), strokewidth = 1.5)
    lines!(real_lr_pixel_value .+ [0,0], [1e-4, 0.3], label = "true value, low res.", color=(:green, 1.5), strokewidth = 1.5)
end
axislegend(; position= :lt, labelsize= 16)

ax = Axis(fig[1,4], xlabel="Pixel value", ylabel="Cumulative Distribution", title="All pixels",titlefont = :regular)
fake_hr_pixel_values = Random.shuffle(fake_hr_samples[:, :, ch,:])[1:512*512]
real_hr_pixel_values = real_hr_sample[:,:, ch,1][:]
real_lr_pixel_values = real_lr_sample[:,:, ch,1][:]
cdf_train_hr = cdf.(x, Ref(real_hr_pixel_values))
cdf_gen_hr = cdf.(x, Ref(fake_hr_pixel_values))
cdf_train_lr = cdf.(x, Ref(real_lr_pixel_values))

lines!(x, cdf_train_hr, color=(:orange, 1.0), strokewidth = 1.5,label="true high res.")
lines!(x, cdf_train_lr, color=(:green, 1.0), strokewidth = 1.5,label="true low res.")
lines!(x, cdf_gen_hr, color=(:purple, 1.0), strokewidth = 1.5, label="gen. high res.")
ks_gen_hr_train_hr = maximum(abs.(cdf_train_hr .- cdf_gen_hr))
ks_train_lr_train_hr = maximum(abs.(cdf_train_hr .- cdf_train_lr))

@info ks_gen_hr_train_hr
@info ks_train_lr_train_hr
ylims!(ax, 0, 1.0)
xlims!(ax, min_x, max_x)

axislegend(; position= :lt, labelsize= 16)
save("fig:paired_pixels_ch$(ch)_v2.png", fig, px_per_unit = 2)





