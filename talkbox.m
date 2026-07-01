%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% HW1 - TalkBox Effect                     
% DAAP course -- NG
% 29/06/2026
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear; close all;

%% Parameters *************************************************************
% Windowing
cfg.frame_len_ms    = 20;       % window length [ms]
cfg.overlap         = 0.5;      % windowing overlap ratio

% LPC order
cfg.P_low           = 48;       % modulator order for low frequencies
cfg.P_high          = 44;       % modulator order for high frequencies 
cfg.P_x             = 48;       % carrier order

stereo_delay_ms     = 26;       % stereo delay [ms]


%% Processing *************************************************************
[s_mod, fs_s] = audioread("SAMPLES/Toms_diner.wav");     % load modulator signal (speech)
[s_car, fs_x] = audioread("SAMPLES/moore_guitar.mp3");   % load carrier signal = excitation source (instrument)

% Preprocess and synchronize modulator and carrier signals
[car_sync, mod_sync, fs] = preprocess_audio_signals(s_car, fs_x, s_mod, fs_s);

% Execute dual-band LPC cross-synthesis to generate talkbox effect:
% shapes the carrier excitation source with the modulator spectral envelope
[y, diag] = lpc_talkbox_dualband_engine(car_sync, mod_sync, fs, cfg);

t = (0:length(y)-1)/fs; % time vector

% Compute the modulator spectrum for visualization
mod_frame = diag.mod_aligned((floor(length(diag.mod_aligned)/2)) + (1:diag.M_samples)) .* hamming(diag.M_samples);
mod_fft_db = db(abs(fft(mod_frame, diag.M_samples))) - max(db(abs(fft(mod_frame, diag.M_samples))));



%% Visualization **********************************************************
figure(1); 
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.05 0.05 0.9 0.9]);
tiledlayout(3, 4, 'TileSpacing', 'compact', 'Padding', 'compact');

% Subplot 1 -- time domain signals comparison: modulator vs talkbox output
nexttile([1 3]); 
h1 = plot(t, diag.mod_aligned/max(abs(diag.mod_aligned)), 'Color', [0.2 0.7 0.3], 'LineWidth', 1.8); 
hold on; 
h2 = plot(t, y/max(abs(y)), 'Color', [0 0 0], 'LineWidth', 1.8); hold off;
grid on; ylabel('Normalized Amplitude'); xlabel('Time [s]');
title('Modulator vs reconstructed signals'); legend([h1, h2], {'Modulator (speech)', 'Talkbox output'});

% Text panel : shows system parameters
nexttile(4, [2 1]); axis off;
text(0.1, 0.9, {'\bf{SYSTEM PARAMETERS}', '', sprintf('Window length: %.1f ms = %d samples', diag.M_ms, diag.M_samples), ...
    sprintf('Hop size: %.0f%% = %d samples', cfg.overlap*100,  diag.hop_samples), '', '\bf{LPC ORDERS}', ...
    sprintf('Car: %d, Low: %d, High: %d', cfg.P_x, cfg.P_low, cfg.P_high)}, 'FontSize', 11, 'Interpreter', 'tex');

% Subplot 2 -- time domain signals comparison: carrier vs talkbox output
nexttile([1 3]); 
h3 = plot(t, diag.car_aligned/max(abs(diag.car_aligned)), 'Color', [0.9 0.5 0.1], 'LineWidth', 1.8);
hold on; 
h4 = plot(t, y/max(abs(y)), 'Color', [0 0 0], 'LineWidth', 1.8); hold off;
grid on; ylabel('Normalized Amplitude'); xlabel('Time [s]');
title('Carrier vs reconstructed signals'); legend([h3, h4], {'Carrier (instrument)', 'Talkbox output'});

% Subplot 3 -- Spectral envelope comparison: LPC envelope vs modulator vs talkbox output
nexttile([1 3]); 
h5 = plot(diag.wlags/1000, diag.H_s_db, 'Color', [1 0 0.9], 'LineWidth', 1.8);
hold on; 
h6 = plot(diag.f_axis/1000, mod_fft_db, 'Color', [0.2 0.7 0.3], 'LineWidth', 1.2); 
h7 = plot(diag.f_axis/1000, diag.Y_fft_db, 'Color', [0 0 0], 'LineWidth', 1.2); 
hold off;
xlim([0 fs/2000]); grid on; ylabel('Magnitude [dB]'); xlabel('Frequency [kHz]');
title('Spectral envelopes'); legend([h5, h6, h7], {'LPC envelope', 'Modulator spectrum', 'Talkbox output spectrum'});


%% Output and file write **************************************************
y_mono = y / max(abs(y));                       % normalize mono output signal
del = round(stereo_delay_ms * fs / 1000);       % compute stereo delay in samples
y_stereo = [y_mono, [zeros(del, 1); y_mono(1:end-del)]]; % write stero arrays, left no delay, right delayed

filename = sprintf('OUTPUTS/output_car%d_mlow%d_mhigh%d_M%d_hop%d.wav', ...
    cfg.P_x, cfg.P_low, cfg.P_high, diag.M_samples, diag.hop_samples); % dynamic filename

% audiowrite(filename, y_stereo, fs);             % write and save audio file
player = audioplayer(y_stereo, fs);
play(player);                                   % play audio



%% LOCAL FUNCTIONS ********************************************************

% PRE-PROCESSING **********************************************************

function [car_out, mod_out, fs] = preprocess_audio_signals(car, fs_car, mod, fs_mod)
% PREPROCESS_AUDIO_SIGNALS : downsamples, makes mono, truncates and normalises arrays
% Input signals are expected to be mono

    % Select lower sample rate
    fs = min(fs_car, fs_mod); 
    
    % Force mono if stereo, resample to common sample rate
    car = resample(car(:,1), fs, fs_car); 
    mod = resample(mod(:,1), fs, fs_mod);
    
    % Synchronize signal arrays
    N = min(length(car), length(mod));      % minimum array length
    mod_out = mod(1:N) / max(abs(mod));     % truncate and normalise modulator signal
    car_out = car(1:N) / max(abs(car));     % truncate and normalise carrier signal
    
end



% LPC ENGINE **************************************************************

function [y, diag] = lpc_talkbox_dualband_engine(car, mod, fs, cfg)
% LPC_TALKBOX_DUALBAND_ENGINE: Runs dual-band frame LPC convolution using MATLAB lpc function
%
% - segment audio into overlapping frames
% - split modulator into low and high frequency crossover bands
% - run lpc analysis to get carrier whitening filter and modulator shaping filters
% - filter carrier to extract the prediction error 
% - resynthesize signal by shaping residual with modulator shaping filters
% - clamp local frame peaks and perform OLA synthesis
% - capture a middle frame spectral snapshot for figure diagnostics
% - apply a 30 ms moving average window to bound global output amplitude

    N = length(car);                        % total number of samples to process
    
    % Frame Analysis Parameters
    M = 2^nextpow2(floor(cfg.frame_len_ms * fs / 1000));  % next power of 2 window length
    hop = floor(M * (1 - cfg.overlap));     % frame hop size

    w_ana = hamming(M);                     % analysis windowing function
    w_syn = hann(M);                        % synthesis windowing function

    N_frames = floor((N - M) / hop);        % total number of frames to iterate
    y = zeros(N + M, 1);                    % preallocate output signal vector

    % Crossover Filters
    % Split modulator spectra into low and high frequency bands at 1 kHz
    [b_l, a_l] = butter(2, 1000 / (fs/2), 'low');   % 2nd-order lowpass crossover filter
    [b_h, a_h] = butter(2, 1000 / (fs/2), 'high');  % 2nd-order highpass crossover filter


    % Frame by frame processing loop
    for m = 1:N_frames
        idx = (m-1)*hop + (1:M);            % sample indices for current frame
        s_f = mod(idx) .* w_ana;            % apply analysis window to modulator frame
        x_f = car(idx);                     % extract unwindowed carrier frame
        
        % LPC analysis on modulator 
        % estimate the coefficients of the SHAPING filter 1/A(z) and frame gains
        [a_mod_low, g_l]  = lpc(filter(b_l, a_l, s_f), cfg.P_low);   % low-band modulator shaping filter coefficients
        [a_mod_high, g_h] = lpc(filter(b_h, a_h, s_f), cfg.P_high);  % high-band modulator shaping filter coefficients
        
        % LPC Analysis on carrieR
        % estimate the coefficients of the WHITENING filter A(z)
        [a_car, g_x]      = lpc(x_f, cfg.P_x);                       % carrier whitening filter coefficients

        % Threshold-based noise gate on modulator frame energy
        if rms(mod(idx)) < 0.01
            g_l = 0; 
            g_h = 0; 
        end           
        
        % Compute the prediction error & cross-synthesis
        % Extract residual by applying whitening filter A(z) to carrier
        e = filter(a_car, 1, x_f);                                
        
        % Reconstruct signal by applying shaping filters 1/A(z) to the residual
        ym = filter(sqrt(g_l/(g_x+eps)), a_mod_low, e) + filter(sqrt(g_h/(g_x+eps)), a_mod_high, e); 
        
        % Dynamic envelope clamping
        car_peak = max(abs(x_f));                % peak amplitude of the carrier frame
        ym_peak  = max(abs(ym));                 % peak amplitude of the synthesized talkbox frame

        if ym_peak > car_peak && ym_peak > 0
            ym = ym * (car_peak / ym_peak);      % scale down synthesized frame to prevent clipping
        end  
        
        y(idx) = y(idx) + (ym .* w_syn);         % overlap-and-add using synthesis window


        
        % Capture a single frame spectral envelope snapshot for visualization (subplot 3)        
        if m == floor(N_frames / 2)    
            
            % Save frame parameters
            diag.M_samples = M; 
            diag.hop_samples = hop; 
            diag.M_ms = (M/fs) * 1000;
            
            % Compute frequency response of the low-band and high-band shaping filters
            H_c = freqz(sqrt(g_l), a_mod_low, M, fs) + freqz(sqrt(g_h), a_mod_high, M, fs);
            
            % Generate linear frequency axis for the lpc filter response plot
            diag.wlags = (0:M-1)*(fs/(2*M)); 
            
            % Normalize the shaping filter magnitude spectrum to 0 db
            diag.H_s_db = db(abs(H_c)) - max(db(abs(H_c)));
            
            % Generate linear frequency axis for the output fft plot
            diag.f_axis = (0:M-1) * (fs / M);
            
            % Compute output fft and normalize to 0 db
            diag.Y_fft_db = db(abs(fft(ym .* w_syn, M))) - max(db(abs(fft(ym .* w_syn, M))));
        end

    end
    
    % Smooth and clamp output amplitude: output must never exceed modulator or carrier boundaries        
    y = y(1:N);      % truncate tail due to linear OLA                     
    
    % Moving average filter to track average energy of 30 ms window
    env_w = ones(floor(0.03*fs), 1) / floor(0.03*fs); 
    
    % Bound output amplitude underneath input signal envelopes
    y = y .* min(1, min(filter(env_w,1,abs(mod)), filter(env_w,1,abs(car))) ./ (filter(env_w,1,abs(y)) + eps));
    
    % Save signals  for plotting 
    diag.mod_aligned = mod;                 
    diag.car_aligned = car;                 
                 
end



