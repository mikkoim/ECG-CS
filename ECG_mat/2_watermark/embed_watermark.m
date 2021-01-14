function [bw, v] = embed_watermark(w, B, y, params)
em_power = params.em_power;

BT = pinv(B);

bw = B*w; % Embed watermark
bw = bw./norm(bw); % Normalization

alpha = norm(y).*(em_power);
bw = bw.*alpha;

bw_in = BT*bw; % Inverse embedding
v = abs(bw_in(1));
end

