function [fval, x] = lpipm(c,A,b)
%input:
%c is coefficient vector of object function
%A is coefficient matrix
%b is right hand side value

[m,n] = size(A);
%initialization
[x,y,s] = deal(ones(n,1), zeros(m,1), ones(n,1));
tol = 1e-7;
[xd, xp, xc] = deal(c-A'*y-s,b-A*x,-x.*s); % r.h.s of newton system
while (max([norm(xp)/(1+norm(b)); norm(xd)/(1+norm(c)); x'*s/(n)]) > tol)
N = sparse(A*spdiags(x./s,0,n,n)*A');  %normal eqaution
[L, ~, S] = chol(N, 'lower');
[dx1, dy1, ds1] = direction(xd, xp, xc, L, S, A, x, s); %predictor dirction
[ap, ad] = deal(step(x./dx1), step(s./ds1)); 
mu = ((x+ap*dx1)'*(s+ad*ds1))^3/(m*(x'*s)^2); 
[dx2, dy2, ds2] = direction(xd*0, xp*0, mu-dx1.*ds1, L, S, A, x, s); %corrector direction
[ap, ad] = deal(step(x./(dx1+dx2)),step(s./(ds1+ds2)));
[x,y,s] = deal(x+.99*ap*(dx1+dx2), y+.99*ad*(dy1+dy2), s+.99*ad*(ds1+ds2));
[xd, xp, xc] = deal(c-A'*y-s,b-A*x,-x.*s);
end
fval = c'*x;
end
function alpha = step(d)  %find the step for dual or primal problem
alpha = (isempty(d(:)<0))+(~isempty(d(:)<0))*(min(-0.9*d(d(:)<0)));
end
function [dx, dy, ds] = direction(xd, xp, xc, L, S, A, x, s)
xi = xp + A*(x.*(xd - xc./x)./s); % r.h.s of the normal equation
dy = S*(L'\(L\(S'*xi))); 
ds = xd - A'*dy;
dx = (xc - x.*ds)./s;
end
