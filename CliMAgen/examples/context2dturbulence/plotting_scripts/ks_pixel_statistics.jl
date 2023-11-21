channels = [1, 2]
wavenumbers = [2.0, 4.0, 8.0, 16.0]
output_ch1 = zeros((4,4))
output_ch2 = zeros((4,4))
for ch in channels
    @show ch
    fig = Figure(resolution=(1600, 400), fontsize=24)
    # Set min and max x values based on channel
    min_x, max_x = ch == 1 ? (-25, 5) : (-20, 20)
    X = LinRange(min_x, max_x, 1000)
    # Loop through each wavenumber
    for i in 1:4
        wn = wavenumbers[i]
        # Read high and low resolution training and generated data
        train_hr = readdlm(joinpath(hr_stats_basedir, "train/train_pixels_ch$(ch)_$(wn).csv"), ',', Float32, '\n')[:]
        train_lr = readdlm(joinpath(lr_stats_basedir, "train/train_pixels_ch$(ch)_0.0.csv"), ',', Float32, '\n')[:]
        gen_hr = readdlm(joinpath(hr_stats_basedir, "downscale_gen/downscale_gen_pixels_ch$(ch)_$(wn).csv"), ',', Float32, '\n')[:]

        # Calculate cdfs for the training and generated data
        cdf_train_hr = cdf.(X, Ref(train_hr))
        cdf_gen_hr = cdf.(X, Ref(gen_hr))
        cdf_train_lr = cdf.(X, Ref(train_lr))

        # Calculate the maximum differences for KS statistics
        ks_gen_hr_train_hr = maximum(abs.(cdf_train_hr .- cdf_gen_hr))
        ks_train_lr_train_hr = maximum(abs.(cdf_train_hr .- cdf_train_lr))

        # Add to figure
        # Set axis labels based on channel and create axis
        ax_label = ch == 1 ? "Supersaturation" : "Vorticity"
        ax = Axis(fig[1, i], xlabel=ax_label, ylabel="CDF", title=L"k_x = k_y = %$(Int(wavenumbers[i]))")
        lines!(X, cdf_train_hr, color=(:orange, 1.0), strokewidth = 1.5,label="real high res.")
        lines!(X, cdf_train_lr, color=(:green, 1.0), strokewidth = 1.5, label="generated high res.")
        lines!(X, cdf_gen_hr, color=(:purple, 1.0), strokewidth = 1.5, label="real low res.")
        xlims!(ax, min_x, max_x)
        ylims!(ax, 0, 1)

        # Save results to output matrices
        if ch==1
            output_ch1[i,:] .= [ch, wn, ks_gen_hr_train_hr, ks_train_lr_train_hr]
        elseif ch==2
            output_ch2[i,:] .= [ch, wn, ks_gen_hr_train_hr, ks_train_lr_train_hr]
        end

    end
    axislegend(; position=:lt, labelsize=16)
    save("fig:cdf_ch$(ch).png", fig, px_per_unit= )
end

columns = ["Channel" "Wavenumber"  "KS-DGHR-THR" "KS-TLR-THR"]
output_data = vcat(columns,output_ch1, output_ch2)
open("ks_stats.txt", "w") do io
    writedlm(io, output_data,',')
end
