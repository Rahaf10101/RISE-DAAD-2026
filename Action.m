function S=Action(theta,beta)
% T=1; X=2;
S=beta*sum(real(1-PlaquetteTest(theta,1,2)),'all');
end