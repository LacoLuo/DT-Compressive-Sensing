%% Keep the users with paths
num_user = numel(DeepMIMO_dataset{1, 1}.user);
all_LoS = zeros(num_user,1);

for u=1:num_user
    user_LoS = DeepMIMO_dataset{1, 1}.user{1, u}.LoS_status;
    all_LoS(u) = user_LoS;
end
user_with_path = find(all_LoS~=-1);
num_user_with_path = numel(user_with_path);
all_LoS = all_LoS(user_with_path);

%% Extract channels and positions
channel_shape = size(DeepMIMO_dataset{1, 1}.user{1,1}.channel, 1, 2);
all_channel = zeros([num_user_with_path, channel_shape]);
all_pos = zeros([num_user_with_path, 3]);

for u_=1:num_user_with_path
    u = user_with_path(u_);
    
    user_channel = DeepMIMO_dataset{1, 1}.user{1, u}.channel; % (rx, tx, paths)
    user_pos = DeepMIMO_dataset{1, 1}.user{1, u}.loc;

    if ndims(user_channel) > 2
        user_channel = sum(user_channel, 3);
        all_channel(u_, :, :) = single(user_channel);
    else
        all_channel(u_, :, :) = single(user_channel);
    end

    all_pos(u_, :) = user_pos;
end

all_channel = single(all_channel);
all_pos = single(all_pos);
all_LoS = single(all_LoS);

%% Find the optimal beam indices
N_BS = channel_shape(1, 2);
CB = UPA_codebook_generator_DFT(N_BS, 1, 1, 1, 1, 1, .5);

all_beam_idx = zeros(num_user_with_path, 1);
for u = 1:num_user_with_path
    user_channel = all_channel(u, :);
    gain = abs(user_channel * conj(CB));
    [~, beam_idx] = max(gain, [], 2);
    all_beam_idx(u) = beam_idx;
end

output_dir = "Datasets\Boston5G_3p5_nofoliage_shifted_1\";
if ~exist(output_dir, 'dir')
   mkdir(output_dir)
end

save(output_dir+"dataset.mat", "all_beam_idx", "all_channel")


