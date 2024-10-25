load("results\Boston5G_3p5_nofoliage_shifted_1\all_avg_acc_train_on_DT_32x1_finetune.mat")
all_avg_acc_train_on_DT_shifted_1_finetune = mean(all_avg_acc_train_on_DT, 2);
load("results\Boston5G_3p5_nofoliage_shifted_2\all_avg_acc_train_on_DT_32x1_finetune.mat")
all_avg_acc_train_on_DT_shifted_2_finetune = mean(all_avg_acc_train_on_DT, 2);

load("results\Boston5G_3p5_target\all_avg_acc_train_on_real_num_data.mat")
all_avg_acc_train_on_real = mean(all_avg_acc_train_on_real, 2);

dict_size = [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240];

set_default_plot();

figure;
semilogx(dict_size, all_avg_acc_train_on_real, '-s', MarkerSize=14, MarkerFaceColor='white', Color="#464145");
hold on
semilogx(dict_size, all_avg_acc_train_on_DT_shifted_1_finetune, '-s', MarkerSize=14, MarkerFaceColor='white', Color="#f2606a");
hold on
semilogx(dict_size, all_avg_acc_train_on_DT_shifted_2_finetune, '-s', MarkerSize=14, MarkerFaceColor='white', Color="#78d7dd");
hold on
yline(0.9547, '--', Color="#f2606a", LineWidth=2)
hold on
yline(0.9218, '--', Color="#78d7dd", LineWidth=2)

grid on
box on

legend('Trained on target (real) data', ...
       'Pretrained on DT synthetic data (1-meter error)', ...
       'Pretrained on DT synthetic data (2-meter error)', Location='southeast');

plot([NaN NaN], [NaN NaN], Color='k', LineStyle="--", DisplayName='Before refinement', LineWidth=2)

set(gca, 'LooseInset', get(gca, 'TightInset'));

xlim([10, 10240])
xlabel('Number of target data points');
ylabel('Accuracy');