load("results\Boston5G_3p5_target/all_avg_acc_train_on_real_32x1_dict_size.mat")
all_avg_acc_train_on_real_32x1 = mean(all_avg_acc_train_on_real, 2);
load("results\Boston5G_3p5_nofoliage_shifted_1/all_avg_acc_train_on_DT_32x1_dict_size.mat")
all_avg_acc_train_on_DT_shifted_1_32x1 = mean(all_avg_acc_train_on_DT, 2);
load("results\Boston5G_3p5_nofoliage_shifted_2/all_avg_acc_train_on_DT_32x1_dict_size.mat")
all_avg_acc_train_on_DT_shifted_2_32x1 = mean(all_avg_acc_train_on_DT(:,1:10), 2);

dict_size = [1, 2, 4, 8, 16, 32];

set_default_plot();

figure;
plot(dict_size, all_avg_acc_train_on_real_32x1, '-s', MarkerSize=14, MarkerFaceColor='white', Color="#464145");
hold on;
plot(dict_size, all_avg_acc_train_on_DT_shifted_1_32x1, '-s', MarkerSize=14, MarkerFaceColor='white', Color="#f2606a");
hold on;
plot(dict_size, all_avg_acc_train_on_DT_shifted_2_32x1, '-s', MarkerSize=14, MarkerFaceColor='white', Color="#78d7dd");

grid on;
box on;

set(gca, 'LooseInset', get(gca, 'TightInset'));

xlim([1,32])
xlabel('Number of measurement vectors');
ylabel('Accuracy');
legend('Trained on target (real) data', ...
       'Trained on DT synthetic data (1-meter error)', ...
       'Trained on DT synthetic data (2-meter error)', Location='southeast');