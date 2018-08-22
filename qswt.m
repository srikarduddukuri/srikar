function [LL,LH,HL,HH,LL1,LH1,HL1,HH1,pos,t,TT,TT1] = qswt(LL,LH,HL,HH,LL1,LH1,HL1,HH1,vect_desend,a,b,t)

[m n] = size(LL);
[o p] = size(LL1);
x = 0;
y = 0;
x2 = 0;
y2 = 0;
x3 = 0;
y3 = 0;
for i = 1:m
    for j = 1:n
        x = x + LH(i,j)^2;
    end
end
for k = 1:o
    for l = 1:p
        y = y + LH1(k,l)^2;
    end
end
Ep1 = x + y;

for i = 1:m
    for j = 1:n
        x2 = x2 + HL(i,j)^2;
    end
end
for k = 1:o
    for l = 1:p
        y2 = y2 + HL1(k,l)^2;
    end
end
Ep2 = x2 + y2;

for i = 1:m
    for j = 1:n
        x3 = x3 + HH(i,j)^2;
    end
end
for k = 1:o
    for l = 1:p
        y3 = y3 + HH1(k,l)^2;
    end
end
Ep3 = x3 + y3;

Ep = [Ep1 Ep2 Ep3];
[~,pos] = max(Ep);
if pos == 1
    S = LH;
    S1 = LH1;
elseif pos == 2
    S = HL;
    S1 = HL1;
elseif pos == 3
    S = HH;
    S1= HH1;
end

xx = 0;
for i = 1:m
    for j = 1:n
        xx = xx + S(i,j);
    end
end
T1 = xx / (i*j);

yy = 0;
for k = 1:o
    for l = 1:p
        yy = yy + S1(k,l);
    end
end
T2 = yy / (k*l); 

Sx = [];
c = 0.5;
TT = [0 0 0];
for i = 1:m
    for j = 1:n
        if t <= a*b
            Sx(i,j) = S(i,j)*(1 + c*vect_desend(t));
            TT = [TT;i j vect_desend(t)];
            t = t+1;
        else
            Sx(i,j) = S(i,j);
        end
    end
end

Sy = [];
c1 = 0.5;
TT1 = [0 0 0];
for i = 1:o
    for j = 1:p
        if S1(i,j) ~= S1(1,1)
            if t <= a*b
            TT1 = [TT1;i j Sx(i,j)];
            Sy(i,j) = S1(i,j)*(1 + c*vect_desend(t));
            t = t+1;
            else
                Sy(i,j) = S1(i,j);
            end
        else
            Sy(i,j) = S1(i,j);
        end
    end
end

if pos == 1
    LH = Sx;
    LH1 = Sy;
elseif pos == 2
    HL = Sx;
    HL1 = Sy;
elseif pos == 3
    HH = Sx;
    HH1 = Sy;
end