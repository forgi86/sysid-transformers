clc;
clear;
close all;

data = load("lin_data.mat");
batch_size = size(data.batch_u, 1);
seq_len = size(data.batch_u, 2);
seq_len_ctx = 400;

err = zeros(batch_size, 1);
y_sim_all = zeros(batch_size, seq_len);

noise_std = 0.0;
for batch_idx=1:batch_size
    u = double(data.batch_u(batch_idx, :));
    u = u(:);
    y = double(data.batch_y(batch_idx, :));
    y = y(:);


    % ident
    y_id = y(1:seq_len_ctx);
    y_id = y_id + randn(size(y_id))*noise_std;
    u_id = u(1:seq_len_ctx);
    idata = iddata(y_id, u_id);
    idata_full = iddata(y, u);

    model = n4sid(idata, 10, "InitialState", "estimate", "focus", "simulation");
    %model = n4sid(idata, [], "InitialState", "estimate", "focus", "simulation"); % estimate state dimension
    %x0 = model.Report.Parameters.X0;

    %y_sim = sim(model, u, "InitialState", x0);
    y_sim = compare(idata_full, model);
    y_sim = y_sim.OutputData;

    y_sim_all(batch_idx, :) = y_sim;
    err(batch_idx) = rmse(y(seq_len_ctx:end), y_sim(seq_len_ctx:end));

end

disp(mean(err))
[z, worst_idx] = max(err);
plot(data.batch_y(worst_idx, :))
hold on
plot(y_sim_all(worst_idx, :), "r")

% 0 noise: 0.0073
% 0.1 noise: 0.0922
% 0.2 noise: 0.1113
% 0.3 noise: 0.1301
% 0.4 noise: 0.1476
% 0.5 noise: 0.1651 