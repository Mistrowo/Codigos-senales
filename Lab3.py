import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import scipy.integrate as integrate

#Exponencial decreciente
t = np.arange(-3,3,0.1)
decres = np.exp(-0.7*t)


plt.subplot(3,4,1)
plt.plot(t,decres)
plt.title('Exponencial Decreciente')

#Exponencial creciente
cres = np.exp(0.7*t)


plt.subplot(3,4,2)
plt.plot(t,cres)
plt.title('Exponencial Creciente')

#Impulso
u0 = np.piecewise(t,t>=0,[1,0])
udt = np.piecewise(t,t>=(0+0.1),[1,0])
impulso = u0 - udt

plt.subplot(3,4,3)
plt.plot(t,impulso)
plt.title('Impulso')

#Escalon
u = lambda t: np.piecewise(t,t>=0,[1,0])

u0 = u(t-0)

plt.subplot(3,4,4)
plt.plot(t,u0)
plt.title('Escalon')

#Sinc
t2=np.arange(-10,10,0.1)
s=np.sinc(t2)

plt.subplot(3,4,5)
plt.plot(t2,s)
plt.title('Sinc')

#------------------------------------------------------------------------
#Seno
t1 = np.arange(-10,10,0.1)
sen = np.sin(t1)

plt.subplot(3,4,9)
plt.plot(t1,sen)
plt.title('Seno')

#Cuadrada
cuad = signal.square(2 * np.pi * 0.2 * t1)

plt.subplot(3,4,10)
plt.plot(t1, cuad) 
plt.title('Cuadrada')

#Triangular
trian = signal.sawtooth(2 * np.pi * 0.2 * t1, 0.5)

plt.subplot(3,4,11)
plt.plot(t1,trian)
plt.title('Triangular')

#Sierra
sierra = signal.sawtooth(2 * np.pi * 0.2 * t1)

plt.subplot(3,4,12)
plt.plot(t1, sierra)
plt.title('Sierra') 
plt.show()


#----------------------------------------------------------------------
#Señal seno convolucionada
#Exp decre
senoexp = signal.convolve(sen, decres , mode='same')

plt.subplot(2,5,1)
plt.plot(t1,senoexp)
plt.title('Seno conv exp decre')

#Exp cres
senoexpcres = signal.convolve(sen, cres , mode='same')

plt.subplot(2,5,2)
plt.plot(t1,senoexpcres)
plt.title('Seno conv exp cres')

#Impulso
Imp = signal.convolve(sen, impulso , mode='same')

plt.subplot(2,5,3)
plt.plot(t1,Imp)
plt.title('Seno conv Impulso')

#Escalon
esc = signal.convolve(sen, u0, mode='same')

plt.subplot(2,5,4)
plt.plot(t1,esc)
plt.title('Seno conv Escalon')

#Sinc
sinc = signal.convolve(sen, s, mode='same')

plt.subplot(2,5,5)
plt.plot(t1,sinc)
plt.title('Seno conv Sinc')

#Señal cuadrada convolucionada
#Exp decre
cuadexp = signal.convolve(cuad, decres , mode='same')

plt.subplot(2,5,6)
plt.plot(t1,cuadexp)
plt.title('Cuadrada conv exp decre')

#Exp cres
cuadexpcres = signal.convolve(cuad, cres , mode='same')

plt.subplot(2,5,7)
plt.plot(t1,cuadexpcres)
plt.title('Cuadrada conv exp cres')

#Impulso
cuadImp = signal.convolve(cuad, impulso , mode='same')

plt.subplot(2,5,8)
plt.plot(t1,cuadImp)
plt.title('Cuadrada conv Impulso')

#Escalon
cuadesc = signal.convolve(cuad, u0, mode='same')

plt.subplot(2,5,9)
plt.plot(t1,cuadesc)
plt.title('Cuadrada conv Escalon')

#Sinc
cuadsinc = signal.convolve(cuad, s, mode='same')

plt.subplot(2,5,10)
plt.plot(t1,cuadsinc)
plt.title('Cuadrada conv Sinc')

plt.show()

#Señal triangular convolucionada
#Exp decre
trianexp = signal.convolve(trian, decres , mode='same')

plt.subplot(2,5,1)
plt.plot(t1,trianexp)
plt.title('Triangular conv exp decre')

#Exp cres
trianexpcres = signal.convolve(trian, cres , mode='same')

plt.subplot(2,5,2)
plt.plot(t1,trianexpcres)
plt.title('Triangular conv exp cres')

#Impulso
trianImp = signal.convolve(trian, impulso , mode='same')

plt.subplot(2,5,3)
plt.plot(t1,trianImp)
plt.title('Triangular conv Impulso')

#Escalon
trianesc = signal.convolve(trian, u0, mode='same')

plt.subplot(2,5,4)
plt.plot(t1,trianesc)
plt.title('Seno conv Escalon')

#Sinc
triansinc = signal.convolve(trian, s, mode='same')

plt.subplot(2,5,5)
plt.plot(t1,triansinc)
plt.title('Triangular conv Sinc')

#Señal sierra convolucionada
#Exp decre
sierraexp = signal.convolve(sierra, decres , mode='same')

plt.subplot(2,5,6)
plt.plot(t1,sierraexp)
plt.title('Sierra conv exp decre')

#Exp cres
sierraexpcres = signal.convolve(sierra, cres , mode='same')

plt.subplot(2,5,7)
plt.plot(t1,sierraexpcres)
plt.title('Sierra conv exp cres')

#Impulso
sierraImp = signal.convolve(sierra, impulso , mode='same')

plt.subplot(2,5,8)
plt.plot(t1,sierraImp)
plt.title('Sierra conv Impulso')

#Escalon
sierraesc = signal.convolve(sierra, u0, mode='same')

plt.subplot(2,5,9)
plt.plot(t1,sierraesc)
plt.title('Sierra conv Escalon')

#Sinc
sierrasinc = signal.convolve(sierra, s, mode='same')

plt.subplot(2,5,10)
plt.plot(t1,sierrasinc)
plt.title('Sierra conv Sinc')

plt.show()

#-------------------------------------------------------------------------------------
#Convolucion creciente con impulso
creciente = signal.convolve(cres, impulso, mode='same')

plt.subplot(2,3,1)
plt.plot(t,creciente)
plt.title('Creciente conv Impulso')

#Convolucion impulso con escalon
impesc = signal.convolve(impulso, u0, mode='same')

plt.subplot(2,3,2)
plt.plot(t,impesc)
plt.title('Impulso conv Escalon')

#Convolucion escalon con sinc
escsinc = signal.convolve(u0, s, mode='same')

plt.subplot(2,3,3)
plt.plot(t,escsinc)
plt.title('Escalon conv sinc')

#Convolucion sinc con creciente
sinccres = signal.convolve(s, cres, mode='same')

plt.subplot(2,3,4)
plt.plot(t2,sinccres)
plt.title('Sinc conv creciente')

#Convolucion decreciente con creciente
decrescien = signal.convolve(decres, cres, mode='same')

plt.subplot(2,3,5)
plt.plot(t,decrescien)
plt.title('Decreciente conv creciente')

plt.show()


#----------------------------------------------------------------------
#Respuesta al impulso comprobar si es LTI
#Exponencial decreciente
decresmult = decres*5


plt.subplot(3,4,1)
plt.plot(t,decresmult)
plt.title('Exponencial Decreciente multiplicada')

#Exponencial creciente
cresmult = cres


plt.subplot(3,4,2)
plt.plot(t,cresmult)
plt.title('Exponencial Creciente multiplicada')

#Impulso
impulsomult = impulso

plt.subplot(3,4,3)
plt.plot(t,impulsomult)
plt.title('Impulso multiplicado')

#Escalon
umult = u0*5

plt.subplot(3,4,4)
plt.plot(t,umult)
plt.title('Escalon multiplicado')

#Sinc
sincmult = s*5

plt.subplot(3,4,5)
plt.plot(t2,sincmult)
plt.title('Sinc multiplicado')
plt.show()

#-------------------------------------------------------------------
#Exponencial decreciente
tc = np.arange(-3,3,0.1)
decres1 = np.exp(-0.7*(tc-2))


plt.subplot(3,4,1)
plt.plot(tc,decres1)
plt.title('Exponencial Decreciente corrida')

#Exponencial creciente
cres1 = np.exp(0.7*(tc-2))


plt.subplot(3,4,2)
plt.plot((tc-5),cres1)
plt.title('Exponencial Creciente corrida')

#Impulso
u02 = np.piecewise(tc,tc-2>=0,[1,0])
udt1 = np.piecewise(tc,tc-2>=(0+0.1),[1,0])
impulso1 = u02 - udt1

plt.subplot(3,4,3)
plt.plot(tc,impulso1)
plt.title('Impulso corrido')

#Escalon
u = lambda tc: np.piecewise(tc,tc-2>=0,[1,0])

u01 = u(tc)

plt.subplot(3,4,4)
plt.plot(tc,u01)
plt.title('Escalon corrido')

#Sinc5
tc2=np.arange(-10,10,0.1)
s1=np.sinc(tc2-2)

plt.subplot(3,4,5)
plt.plot(tc2,s1)
plt.title('Sinc corrido')
plt.show()

#---------------------------------------------------------------
#Energia y potencia
#Aperiodicas
T  = 2*np.pi/1
cuadrado = decres**2
energia = integrate.simps(cuadrado,t)
potencia = (1/T)*energia
print("La señal aperiodica exponencial decreciente tiene una energia de "+str(energia)+" y la potencia es "+str(potencia))

cuadrado = cres**2
energia = integrate.simps(cuadrado,t)
potencia = (1/T)*energia
print("La señal aperiodica exponencial creciente tiene una energia de "+str(energia)+" y la potencia es "+str(potencia))

cuadrado = impulso**2
energia = integrate.simps(cuadrado,t)
potencia = (1/T)*energia
print("La señal aperiodica impulso tiene una energia de "+str(energia)+" y la potencia es "+str(potencia))

cuadrado = u0**2
energia = integrate.simps(cuadrado,t)
potencia = (1/T)*energia
print("La señal aperiodica escalon tiene una energia de "+str(energia)+" y la potencia es "+str(potencia))

cuadrado = s**2
energia = integrate.simps(cuadrado,t2)
potencia = (1/T)*energia
print("La señal aperiodica sinc tiene una energia de "+str(energia)+" y la potencia es "+str(potencia))
print("---------------------------------------------------------------")
#Periodicas
cuadrado = sen**2
energia = integrate.simps(cuadrado,t1)
potencia = (1/T)*energia
print("La señal periodica seno tiene una energia de "+str(energia)+" y la potencia es "+str(potencia))

cuadrado = cuad**2
energia = integrate.simps(cuadrado,t1)
potencia = (1/T)*energia
print("La señal periodica cuadrada tiene una energia de "+str(energia)+" y la potencia es "+str(potencia))

cuadrado = trian**2
energia = integrate.simps(cuadrado,t1)
potencia = (1/T)*energia
print("La señal periodica triangular tiene una energia de "+str(energia)+" y la potencia es "+str(potencia))

cuadrado = sierra**2
energia = integrate.simps(cuadrado,t1)
potencia = (1/T)*energia
print("La señal periodica sierra tiene una energia de "+str(energia)+" y la potencia es "+str(potencia))
print("---------------------------------------------------------------")
#Convolucion con seno
cuadrado = senoexp**2
energia = integrate.simps(cuadrado,t1)
potencia = (1/T)*energia
print("La señal convolucionada seno x decreciente tiene una energia de "+str(energia)+" y la potencia es "+str(potencia))

cuadrado = senoexpcres**2
energia = integrate.simps(cuadrado,t1)
potencia = (1/T)*energia
print("La señal convolucionada seno x creciente tiene una energia de "+str(energia)+" y la potencia es "+str(potencia))

cuadrado = Imp**2
energia = integrate.simps(cuadrado,t1)
potencia = (1/T)*energia
print("La señal convolucionada seno x impulso tiene una energia de "+str(energia)+" y la potencia es "+str(potencia))

cuadrado = esc**2
energia = integrate.simps(cuadrado,t1)
potencia = (1/T)*energia
print("La señal convolucionada seno x escalon tiene una energia de "+str(energia)+" y la potencia es "+str(potencia))

cuadrado = sinc**2
energia = integrate.simps(cuadrado,t1)
potencia = (1/T)*energia
print("La señal convolucionada seno x sinc tiene una energia de "+str(energia)+" y la potencia es "+str(potencia))
print("---------------------------------------------------------------")
#Convolucion con cuadrada
cuadrado = cuadexp**2
energia = integrate.simps(cuadrado,t1)
potencia = (1/T)*energia
print("La señal convolucionada cuadrada x decreciente tiene una energia de "+str(energia)+" y la potencia es "+str(potencia))

cuadrado = cuadexpcres**2
energia = integrate.simps(cuadrado,t1)
potencia = (1/T)*energia
print("La señal convolucionada cuadrada x creciente tiene una energia de "+str(energia)+" y la potencia es "+str(potencia))

cuadrado = cuadImp**2
energia = integrate.simps(cuadrado,t1)
potencia = (1/T)*energia
print("La señal convolucionada cuadrada x impulso tiene una energia de "+str(energia)+" y la potencia es "+str(potencia))

cuadrado = cuadesc**2
energia = integrate.simps(cuadrado,t1)
potencia = (1/T)*energia
print("La señal convolucionada cuadrada x escalon tiene una energia de "+str(energia)+" y la potencia es "+str(potencia))

cuadrado = cuadsinc**2
energia = integrate.simps(cuadrado,t1)
potencia = (1/T)*energia
print("La señal convolucionada cuadrada x sinc tiene una energia de "+str(energia)+" y la potencia es "+str(potencia))
print("---------------------------------------------------------------")
#Convolucion con triangular
cuadrado = trianexp**2
energia = integrate.simps(cuadrado,t1)
potencia = (1/T)*energia
print("La señal convolucionada triangular x decreciente tiene una energia de "+str(energia)+" y la potencia es "+str(potencia))

cuadrado = trianexpcres**2
energia = integrate.simps(cuadrado,t1)
potencia = (1/T)*energia
print("La señal convolucionada triangular x creciente tiene una energia de "+str(energia)+" y la potencia es "+str(potencia))

cuadrado = trianImp**2
energia = integrate.simps(cuadrado,t1)
potencia = (1/T)*energia
print("La señal convolucionada triangular x impulso tiene una energia de "+str(energia)+" y la potencia es "+str(potencia))

cuadrado = trianesc**2
energia = integrate.simps(cuadrado,t1)
potencia = (1/T)*energia
print("La señal convolucionada triangular x escalon tiene una energia de "+str(energia)+" y la potencia es "+str(potencia))

cuadrado = triansinc**2
energia = integrate.simps(cuadrado,t1)
potencia = (1/T)*energia
print("La señal convolucionada triangular x sinc tiene una energia de "+str(energia)+" y la potencia es "+str(potencia))
print("---------------------------------------------------------------")
#Convolucion con sierra
cuadrado = sierraexp**2
energia = integrate.simps(cuadrado,t1)
potencia = (1/T)*energia
print("La señal convolucionada sierra x decreciente tiene una energia de "+str(energia)+" y la potencia es "+str(potencia))

cuadrado = sierraexpcres**2
energia = integrate.simps(cuadrado,t1)
potencia = (1/T)*energia
print("La señal convolucionada sierra x creciente tiene una energia de "+str(energia)+" y la potencia es "+str(potencia))

cuadrado = sierraImp**2
energia = integrate.simps(cuadrado,t1)
potencia = (1/T)*energia
print("La señal convolucionada sierra x impulso tiene una energia de "+str(energia)+" y la potencia es "+str(potencia))

cuadrado = sierraesc**2
energia = integrate.simps(cuadrado,t1)
potencia = (1/T)*energia
print("La señal convolucionada sierra x escalon tiene una energia de "+str(energia)+" y la potencia es "+str(potencia))

cuadrado = sierrasinc**2
energia = integrate.simps(cuadrado,t1)
potencia = (1/T)*energia
print("La señal convolucionada sierra x sinc tiene una energia de "+str(energia)+" y la potencia es "+str(potencia))
print("---------------------------------------------------------------")
#Convolucion de las aperiodicas
cuadrado = creciente**2
energia = integrate.simps(cuadrado,t)
potencia = (1/T)*energia
print("La señal convolucionada creciente x impulso tiene una energia de "+str(energia)+" y la potencia es "+str(potencia))

cuadrado = impesc**2
energia = integrate.simps(cuadrado,t)
potencia = (1/T)*energia
print("La señal convolucionada impulso x escalon tiene una energia de "+str(energia)+" y la potencia es "+str(potencia))

cuadrado = escsinc**2
energia = integrate.simps(cuadrado,t)
potencia = (1/T)*energia
print("La señal convolucionada escalon x sinc tiene una energia de "+str(energia)+" y la potencia es "+str(potencia))

cuadrado = sinccres**2
energia = integrate.simps(cuadrado,t1)
potencia = (1/T)*energia
print("La señal convolucionada sinc x creciente tiene una energia de "+str(energia)+" y la potencia es "+str(potencia))

cuadrado = decrescien**2
energia = integrate.simps(cuadrado,t)
potencia = (1/T)*energia
print("La señal convolucionada decreciente x creciente tiene una energia de "+str(energia)+" y la potencia es "+str(potencia))