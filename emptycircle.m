function circles = emptycircle(x,y,r,c)
hold on
th = 0:pi/50:2*pi;
x_circle = r * cos(th) + x;
y_circle = r * sin(th) + y;
circles = plot(x_circle, y_circle, c);
%fill(x_circle, y_circle, c)
hold off
axis equal
end
%circleout = circle(3, 4, 2, 'g')            % Call ‘circle’ To Create Green Circle