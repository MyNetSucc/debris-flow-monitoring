%% ========= 使用者選檔 =========
[fvid, p] = uigetfile('*.mp4','選擇輸出影片');
if isequal(fvid,0), return; end
vidFile = fullfile(p, fvid);

[fcsv, ~] = uigetfile('*.csv','選擇對應 CSV', p);
if isequal(fcsv,0), return; end
csvFile = fullfile(p, fcsv);

%% ========= 讀影片 (取 FPS → 轉秒) =========
vObj = VideoReader(vidFile);      % R2022a 內建
fps  = vObj.FrameRate;

%% ========= 讀 CSV =========
opts = detectImportOptions(csvFile ,'TextType','string', ...
                           'VariableNamingRule','modify');
T = readtable(csvFile, opts);

% frame → 秒 (frame0 = 0 s)
tSec = T.frame ./ fps;

%% ========= 燈號歷線圖 =========
alertMap = containers.Map(["green","yellow","red"], [0 1 2]);
yAlert   = arrayfun(@(s) alertMap(s), T.alert);

figure('Name','Alert time-series','Color','w', ...
       'Units','inches','Position',[1 1 16 4]);
stairs(tSec, yAlert,'LineWidth',1.4);
yticks([0 1 2]); yticklabels({'green','yellow','red'});
xlabel('Time (s)'); ylabel('Alert level');
title('Debris-flow alert history');
grid on; box on;
xlim([0 ceil(max(tSec))]);

%% ========= 信心值與 |ΔC| 歷線圖 =========
figure('Name','Confidence time-series','Color','w', ...
       'Units','inches','Position',[1 5 16 5]);
hold on;
plot(tSec, T.river_conf ,'-','LineWidth',1.3);
plot(tSec, T.muddy_conf ,'-','LineWidth',1.3);
plot(tSec, T.debris_flow_conf,'-','LineWidth',1.3);
plot(tSec, T.rock_conf  ,'-','LineWidth',1.3);

% |ΔC|_river（EMA 後 ΔC 取絕對值）
if any(strcmp('river_dC', T.Properties.VariableNames))
    plot(tSec, abs(T.river_dC) ,'--','LineWidth',1.6);
else
    warning('CSV 未包含 river_dC 欄位，略過 |ΔC|_river 曲線');
end
hold off; grid on; box on;

legend({'river\_conf','muddy\_conf','debris\_conf', ...
        'rock\_conf','|ΔC|\_river'},'Location','best');
xlabel('Time (s)'); ylabel('Confidence / |ΔC|');
title('Model confidences & river |ΔC| history');
xlim([0 ceil(max(tSec))]);

%% ========= (選用) 查看原因或存圖 =========
% disp(T.reason);                       % 列出每筆警戒原因
% exportgraphics(gcf,'conf_timeline.png','Resolution',300);

% 如果要額外畫 ΔA，可範例：
% figure;
% plot(tSec, T.river_dA * 100,'LineWidth',1.3);
% ylabel('ΔA (%)'); xlabel('Time (s)');
% title('EMA ΔA – river');
% grid on; box on;
