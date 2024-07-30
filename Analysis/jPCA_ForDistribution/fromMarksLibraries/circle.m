
% circle(center,radius,color,width)
% to make an ellipse:
% circle(center,[radiusX, radiusY],color,width)
function circle(center,radius,color,width)

if length(radius) == 1
    radius = [radius, radius];
end

nums = 0:361;  % overlap slightly

x = center(1) + radius(1) * cos(2*pi*nums/360);
y = center(2) + radius(2) * sin(2*pi*nums/360);


h = plot(x,y,'k');
set(h,'linewidth',width,'color',color);