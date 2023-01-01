%Solution to the acoustic field from a focused piston source
%
%Numerical solution from:
%X Chen, K.Q. Schwarz,  KJ Parker, "Radiation patten of a focused
%transducer: A numerically convergent solution"  JASA 94:2979-2991 1993

%Written by Robin Cleveland, University of Oxford
%December 2012

fname='large_area_0.5Mhz.mat';  % set to a filename if you want to save the field to a .mat file
%fname='test2';

% curvature radius?  63e-3
% aperture width?

f0=0.5e6    %1.1e6;   %Frequency 500000
% f0=1.0e6    %1.1e6;   %Frequency 500000
% f0=2e5    %1.1e6;   %Frequency 500000
% f0=2e6    %1.1e6;   %Frequency 500000
% f0=350000    %1.1e6;   %Frequency 500000
% f0=750000    %1.1e6;   %Frequency 500000
% f0=3e6    %1.1e6;   %Frequency 500000

c0=1488;    %Sound Speed in water
a=30e-3;    %Source Radius  
roc=63e-3;  %Radius of curvature
%Locations to calculate field

%Example for 1D axial field
%x=0.0;
%z=[40:0.2:90]*1e-3;

%Example for 1D radial field
%x=[0.01:0.01:10]*1e-3;
%z=60.1e-3;

%Example for 2D field
% x=[0:0.15:7.5]*1e-3;
% z=[0:0.1:140]*1e-3;

%Example for 2D field
%x=[0:0.1:10]*1e-3;
%z=[40:0.2:90]*1e-3;


%Example for 2D field
x=[0:0.1:20]*1e-3;
z=[0:0.1:140]*1e-3;

%Tolerances for summations
Nmax=40;
termtol=1e-4;

%Calculate some useful parameters
k=2*pi*f0/c0;
ka=k*a;
alpha=asin(a/roc);
G=0.5*k*a*a/roc

p=zeros(length(x),length(z));

%Debugging commands
%nextpc=10;
%    nmax=2;
%n1=0;n2=0;

for zcnt=1:length(z),
    nmax=2;
    for xcnt=1:length(x),
        
r=sqrt(x(xcnt)^2+z(zcnt)^2);
theta=atan(x(xcnt)/z(zcnt));

Y=(ka*a/r)*(1-(r/roc)*cos(theta));
Z=ka*sin(theta);

if abs(Y/Z)<1,   %Shadow
    n=0;
    u1p=besselj(1,Z);
    du1p=u1p;
    while (n<2)|((n<Nmax)&(abs(du1p/u1p)>termtol)),
        n=n+1;
        du1p=(-1)^n*(Y/Z)^(2*n)*besselj(2*n+1,Z);
        u1p=u1p+du1p;
    end
    n1=n;
    
    n=0;
    u2p=(Y/Z)*besselj(2,Z);
    du2p=max(1,u2p);
    while (n<2)|((n<Nmax)&(abs(du2p/u2p))>termtol),
        n=n+1;
        du2p=(-1)^n*(Y/Z)^(2*n+1)*besselj(2*n+2,Z);
        u2p=u2p+du2p;
    end
    n2=n;
    
    IYZ=exp(-i*Y/2)*(Y/Z)*(u1p+i*u2p);
    
    if max([n1 n2])>nmax,
        nmax=max([n1 n2]);
        maxerr=[abs(du1p/u1p) abs(du2p/u2p)];
        xmax=x(xcnt);
        zmax=z(zcnt);
    end
else     %Illuminated
    n=0;
    nu1=(Z/Y)*besselj(1,Z);
    dnu1=nu1;
    while (n<2)|((n<Nmax)&(abs(dnu1/nu1)>termtol)),
        n=n+1;
        dnu1=(-1)^n*(Z/Y)^(2*n+1)*besselj(2*n+1,Z);
        nu1=nu1+dnu1;
    end
    n11=n;   
    
    n=0;
    nu0=besselj(0,Z);
    dnu0=nu0;
    while (n<2)|((n<Nmax)&(abs(dnu0/nu0)>termtol)),
        n=n+1;
        dnu0=(-1)^n*(Z/Y)^(2*n)*besselj(2*n,Z);
        nu0=nu0+dnu0;
    end
    n0=n;
  
    IYZ=-i*exp(i*Z*Z/(2*Y))*(1-exp(-i*0.5*(Y+Z^2/Y))*(nu0+i*nu1));
    
    if max([n0 n11])>nmax,
        nmax=max([n0 n11]);
        maxerr=[abs(dnu0/nu0)  abs(dnu1/nu1)];
        xmax=x(xcnt);
        zmax=z(zcnt);
    end

end

if abs(Y)>termtol,  %Not in focal plane
    p(xcnt,zcnt)=i*exp(-i*k*r)*IYZ/(1-(r/roc)*cos(theta));
else   %In focal plane
    if abs(ka*sin(theta))>0.1,
        %p(xcnt,zcnt)=G*exp(i*k*roc*(1-1/cos(theta)))*besselj(1,ka*sin(theta))/(0.5*ka*tan(theta));
        p(xcnt,zcnt)=i*G*exp(-i*k*roc)*besselj(1,ka*sin(theta))/(0.5*ka*tan(theta));
    else %At focus
        p(xcnt,zcnt)=i*G*exp(-i*k*roc);
    end
end

end
    
%Display output during simulation to show progress
if length(z)==1,
    figure(1)
    plot(x*1e3,abs(squeeze(p)))
elseif length(x)==1,
    figure(2)
    plot(z*1e3,abs(squeeze(p)))
else
    figure(3)
    imagesc(z*1e3,x*1e3,abs(p));
end

%DEBUGGING lines commented out
%Check on how many terms and how accurate solution is
%if nmax>2,
%[nmax xmax zmax]
%maxerr
%end

% %Print out percent complete every 10%
%     thispc=round(100*xcnt/length(x));
%     if thispc>=nextpc,
%         fprintf(1,[num2str(thispc) '%% ']);
%         nextpc=nextpc+10;
%     end
end
% fprintf(1,'\n');

if length(z)==1,
    figure(1)
    plot(x*1e3,abs(squeeze(p)))
elseif length(x)==1,
    figure(2)
    plot(z*1e3,abs(squeeze(p)))
else
    figure(3)
    imagesc(z*1e3,x*1e3,abs(p));
    
        
    pp=[p(end:-1:2,:);p];
    xx=[-x(end:-1:2) x];

%Pressure amplitude    
    figure(4)
    imagesc(z*1e3,xx*1e3,abs(pp));  
    xlabel('Axial (mm)');
    ylabel('Radial (mm)');
    title('Amplitude');
   
%Instantaneous pressure
    t0=roc/c0;  %Time from 
    figure(5)
    imagesc(z*1e3,xx*1e3,real(pp*exp(j*2*pi*f0*t0)));  
    xlabel('Axial (mm)');
    ylabel('Radial (mm)');
    title(['Instantaneous pressure  f0=' num2str(f0/1e6) ' MHz ']);
end

insta_p = real(pp*exp(j*2*pi*f0*t0));
pressure_amp = abs(pp);

if length(fname)>1,
save(fname,'f0','c0','a','roc','x','z','xx','Nmax','termtol','pp','pressure_amp','insta_p');
end




