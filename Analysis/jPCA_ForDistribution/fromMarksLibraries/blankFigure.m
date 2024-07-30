% produces a blank figure with everything turned off
% hf = blankFigure(axLim)
% where axLim = [left right bottom top]
function hf = blankFigure(axLim)

hf = figure; hold on; 
set(gca,'visible', 'off');
set(hf, 'color', [1 1 1]);
axis(axLim); axis square;