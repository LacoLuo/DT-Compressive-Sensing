clearvars
load("results_meas_vecs\real_meas_vecs_M_8.mat")


for i = 1:size(BS_meas_vecs, 2)
    plot_pattern(BS_meas_vecs(:, i));
end