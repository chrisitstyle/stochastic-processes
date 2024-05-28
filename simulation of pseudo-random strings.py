import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter
from scipy.signal import hilbert

# mu = 0     # mean
# sigma = 1  # standard deviation
# N = 100    # sample size
# white_noise_rozklad = np.random.rand(N)
#
# plt.hist(white_noise_rozklad, 100, density=True)  # plot a histogram of Y with # of bins = # of elems in Y
# plt.title('Rozkład')
# plt.xlabel('Indeks próbki')
# plt.ylabel('Wartość')
# plt.grid(True)
# plt.show()


# zadanie 1
# Parametry szumu białego gaussowskiego
mean = 5
std = 10
num_samples = 100000
sampling_rate = 10000  # Przyjmujemy częstotliwość próbkowania równą 1 kHz

# Generowanie szumu białego gaussowskiego
white_noise = np.random.normal(mean, std, num_samples)

# Tworzenie osi czasu w sekundach
time = np.arange(num_samples) / sampling_rate

# Wykres szumu białego Gaussowskiego
plt.figure(figsize=(10, 6))
plt.plot(time, white_noise)
plt.title('Wykres szumu białego Gaussowskiego N(5,10)')
plt.xlabel('Czas (s)')
plt.ylabel('Wartość')
plt.grid(True)
plt.show()

# Tworzenie histogramu
plt.hist(white_noise, bins=100, density=True, color='b', alpha=0.7)

# Dodanie wykresu gęstości prawdopodobieństwa
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-((x - mean) ** 2) / (2 * std ** 2))
plt.plot(x, p, 'k', linewidth=2)

title = f'Szum biały gaussowski N({mean}, {std})'
plt.title(title)
plt.xlabel('Wartość')
plt.ylabel('Prawdopodobieństwo')
plt.legend(['Gęstość prawdopodobieństwa', 'Histogram'])
plt.grid(True)
plt.show()

# zadanie 2.
#obliczanie dystrybuanty
sorted_data = np.sort(white_noise)
y = np.arange(len(sorted_data)) / float(len(sorted_data))

# Wykres dystrubuanty
plt.figure(figsize=(10, 6))
plt.plot(sorted_data, y, color='black', marker='.', linestyle='none')
plt.hist(white_noise, bins=100, density=True,cumulative=True, color='b', alpha=0.7)
plt.title('Dystrybuanta szumu białego Gaussowskiego od 5 do 10')
plt.xlabel('Wartość')
plt.ylabel('Prawdopodobieństwo')
plt.grid(True)
plt.yticks(np.arange(0,1.1,0.1))
plt.show()


# Obliczanie wartości oczekiwanej i wariancji
expected_value = np.mean(white_noise)
variance = np.var(white_noise)
print("")
print("wartość oczekiwana:")
print(expected_value)
print("")
print("wariancja:")
print(variance)

# odchylenie standarowe ze wzoru co podał na tablicy
standard_deviation = np.sqrt(variance)
print("")
print("Odchylenie standardowe:")
print(standard_deviation)

# Obliczanie kowariancji dla wszystkich par punktów danych
covariance_values = []
for lag in range(0, 100):  # zakres opóźnień (lag)
    shifted_data = np.roll(white_noise, lag)
    covariance = np.mean((white_noise - np.mean(white_noise)) * (shifted_data - np.mean(shifted_data)))
    covariance_values.append(covariance)

czas=np.arange(100)/sampling_rate

# Tworzenie wykresu funkcji kowariancyjnej
plt.figure(figsize=(10, 5))
plt.plot(czas, covariance_values, color='blue')
plt.title('Funkcja kowariancyjna dla szumu gaussowskiego')
plt.xlabel('Czas (s)')
plt.ylabel('Wartość kowariancji')
plt.grid(True)
plt.show()



# zadanie 3 i 4 dla Fc=0,05
# Czas
t = np.arange(0, num_samples / sampling_rate, 1 / sampling_rate)

# Parametry filtru
Fc = 0.05  # Częstotliwość odcięcia
F = 500  # Częstotliwość próbkowania sygnału wąskopasmowego
M = 31  # Długość odpowiedzi impulsowej filtra
taps = firwin(M, Fc)

# Filtracja
filtered_signal_005 = lfilter(taps, 1.0, white_noise)

# Wykres sygnału wąskopasmowego
plt.plot(t, filtered_signal_005, 'r-', linewidth=2, label='Sygnał')
plt.title(f'Fc = {Fc}, F = {F} Hz')
plt.xlabel('Czas [s]')
plt.ylabel('Amplituda')
plt.legend()
plt.show()

# Wykres sygnału wąskopasmowego  i gaussowski
plt.plot(t, white_noise, 'b-', label='Sygnał oryginalny')
plt.plot(t, filtered_signal_005, 'r-', linewidth=2, label='Sygnał')
plt.title(f'Fc = {Fc}, F = {F} Hz')
plt.xlabel('Czas [s]')
plt.ylabel('Amplituda')
plt.legend()
plt.show()

# Obliczanie wartości oczekiwanej i wariancji dla filtru 005
expected_value_005 = np.mean(filtered_signal_005)
variance_005 = np.var(filtered_signal_005)
print("")
print("wartość oczekiwana filtru 005:")
print(expected_value_005)
print("")
print("wariancja filtru 005:")
print(variance_005)

# odchylenie standarowe ze wzoru co podał na tablicy
standard_deviation_005 = np.sqrt(variance_005)
print("")
print("Odchylenie standardowe filtru 005:")
print(standard_deviation_005)

# Tworzenie histogramu
plt.hist(filtered_signal_005, bins=100, density=True, color='b', alpha=0.7)

# Dodanie wykresu gęstości prawdopodobieństwa
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 100)
# p = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-((x - mean) ** 2) / (2 * std ** 2))
# plt.plot(x, p, 'k', linewidth=2)

title = f'Filtr Fc=0,05 F=500'
plt.title(title)
plt.xlabel('Wartość')
plt.ylabel('Prawdopodobieństwo')
plt.legend(['Histogram'])
plt.grid(True)
plt.show()

#obliczanie dystrybuanty Filtr Fc=0,05 F=500
sorted_data_005 = np.sort(filtered_signal_005)
y = np.arange(len(sorted_data_005)) / float(len(sorted_data_005))

# Wykres dystrubuanty  Filtr Fc=0,05 F=500
plt.figure(figsize=(10, 6))
plt.plot(sorted_data_005, y, color='black', marker='.', linestyle='none')
plt.hist(filtered_signal_005, bins=100, density=True,cumulative=True, color='b', alpha=0.7)
plt.title('Dystrybuanta filtra Fc=0,05 F=500')
plt.xlabel('Wartość')
plt.ylabel('Prawdopodobieństwo')
plt.grid(True)
plt.yticks(np.arange(0,1.1,0.1))
plt.show()




# Obliczanie kowariancji dla wszystkich par punktów danych
covariance_values_005 = []
for lag in range(0, 100):  # zakres opóźnień (lag)
    shifted_data = np.roll(filtered_signal_005, lag)
    covariance_005 = np.mean((filtered_signal_005 - np.mean(filtered_signal_005)) * (shifted_data - np.mean(shifted_data)))
    covariance_values_005.append(covariance_005)

# Tworzenie wykresu funkcji kowariancyjnej
plt.figure(figsize=(10, 5))
plt.plot(czas, covariance_values_005, color='blue')
plt.title('Funkcja kowariancyjna dla Filtra 005')
plt.xlabel('Czas (s)')
plt.ylabel('Wartość kowariancji')
plt.grid(True)
plt.show()


# Obliczanie pierwszego zera funkcji kowariancji po filtracji
first_zero_index_005 = next((i for i, val in enumerate(covariance_values_005) if val <= 0), None)

if first_zero_index_005 is not None:
    first_zero_delay_005 = first_zero_index_005 * (1 / sampling_rate)  # Opóźnienie dla pierwszego zera, przeliczone na sekundy
    print("Pierwsze zero funkcji kowariancji po filtracji 005:", first_zero_delay_005, "s")
else:
    print("Brak zer w funkcji kowariancji po filtracji 005.")

# zadanie 3 i 4 dla Fc=0,1
# Parametry filtru
Fc = 0.1  # Częstotliwość odcięcia
F = 1000  # Częstotliwość próbkowania sygnału wąskopasmowego
M = 31  # Długość odpowiedzi impulsowej filtra
taps = firwin(M, Fc)

# Filtracja
filtered_signal_01 = lfilter(taps, 1.0, white_noise)

# Wykres sygnału wąskopasmowego
plt.plot(t, filtered_signal_01, 'r-', linewidth=2, label='Sygnał dolnopasmowy')
plt.title(f'Fc = {Fc}, F = {F} Hz')
plt.xlabel('Czas [s]')
plt.ylabel('Amplituda')
plt.legend()
plt.show()

# Wykres sygnału wąskopasmowego i gaussowski
plt.plot(t, white_noise, 'b-', label='Sygnał oryginalny')
plt.plot(t, filtered_signal_01, 'r-', linewidth=2, label='Sygnał dolnopasmowy')
plt.title(f'Fc = {Fc}, F = {F} Hz')
plt.xlabel('Czas [s]')
plt.ylabel('Amplituda')
plt.legend()
plt.show()

# Obliczanie wartości oczekiwanej i wariancji dla filtru 01
expected_value_01 = np.mean(filtered_signal_01)
variance_01 = np.var(filtered_signal_01)
print("")
print("wartość oczekiwana filtru 01:")
print(expected_value_01)
print("")
print("wariancja filtru 01:")
print(variance_01)

# odchylenie standarowe ze wzoru co podał na tablicy
standard_deviation_01 = np.sqrt(variance_01)
print("")
print("Odchylenie standardowe filtru 01:")
print(standard_deviation_01)

# Tworzenie histogramu
plt.hist(filtered_signal_01, bins=100, density=True, color='b', alpha=0.7)

title = f'Filtr Fc=0,1 F=1000'
plt.title(title)
plt.xlabel('Wartość')
plt.ylabel('Prawdopodobieństwo')
plt.legend(['Histogram'])
plt.grid(True)
plt.show()

#obliczanie dystrybuanty Filtr Fc=0,1 F=1000
sorted_data_01 = np.sort(filtered_signal_01)
y = np.arange(len(sorted_data_01)) / float(len(sorted_data_01))

# Wykres dystrubuanty  Filtr Fc=0,1 F=1000
plt.figure(figsize=(10, 6))
plt.plot(sorted_data_01, y, color='black', marker='.', linestyle='none')
plt.hist(filtered_signal_01, bins=100, density=True,cumulative=True, color='b', alpha=0.7)
plt.title('Dystrybuanta filtra Fc=0,1 F=1000')
plt.xlabel('Wartość')
plt.ylabel('Prawdopodobieństwo')
plt.grid(True)
plt.yticks(np.arange(0,1.1,0.1))
plt.show()

# Obliczanie kowariancji dla wszystkich par punktów danych
covariance_values_01 = []
for lag in range(0, 100):  # zakres opóźnień (lag)
    shifted_data = np.roll(filtered_signal_01, lag)
    covariance_01 = np.mean((filtered_signal_01 - np.mean(filtered_signal_01)) * (shifted_data - np.mean(shifted_data)))
    covariance_values_01.append(covariance_01)

# Tworzenie wykresu funkcji kowariancyjnej
plt.figure(figsize=(10, 5))
plt.plot(czas, covariance_values_01, color='blue')
plt.title('Funkcja kowariancyjna dla Filtra 01')
plt.xlabel('Czas (s)')
plt.ylabel('Wartość kowariancji')
plt.grid(True)
plt.show()

# Obliczanie pierwszego zera funkcji kowariancji po filtracji
first_zero_index_01 = next((i for i, val in enumerate(covariance_values_01) if val <= 0), None)

if first_zero_index_01 is not None:
    first_zero_delay_01 = first_zero_index_01 * (1 / sampling_rate)  # Opóźnienie dla pierwszego zera, przeliczone na sekundy
    print("Pierwsze zero funkcji kowariancji po filtracji 01:", first_zero_delay_01, "s")
else:
    print("Brak zer w funkcji kowariancji po filtracji 01.")



# zadanie 3 i 4 dla Fc=0,2
# Parametry filtru
Fc = 0.2  # Częstotliwość odcięcia
F = 2000  # Częstotliwość próbkowania sygnału wąskopasmowego
M = 31  # Długość odpowiedzi impulsowej filtra
taps = firwin(M, Fc)

# Filtracja
filtered_signal_02 = lfilter(taps, 1.0, white_noise)

# Wykres sygnału wąskopasmowego i gaussowski
plt.plot(t, filtered_signal_02, 'r-', linewidth=2, label='Sygnał dolnopasmowy')
plt.title(f'Fc = {Fc}, F = {F} Hz')
plt.xlabel('Czas [s]')
plt.ylabel('Amplituda')
plt.legend()
plt.show()

# Wykres sygnału wąskopasmowego
plt.plot(t, white_noise, 'b-', label='Sygnał oryginalny')
plt.plot(t, filtered_signal_02, 'r-', linewidth=2, label='Sygnał dolnopasmowy')
plt.title(f'Fc = {Fc}, F = {F} Hz')
plt.xlabel('Czas [s]')
plt.ylabel('Amplituda')
plt.legend()
plt.show()

# Obliczanie wartości oczekiwanej i wariancji dla filtru 02
expected_value_02 = np.mean(filtered_signal_02)
variance_02 = np.var(filtered_signal_02)
print("")
print("wartość oczekiwana filtru 02:")
print(expected_value_02)
print("")
print("wariancja filtru 02:")
print(variance_02)

# odchylenie standarowe ze wzoru co podał na tablicy
standard_deviation_02 = np.sqrt(variance_02)
print("")
print("Odchylenie standardowe filtru 02:")
print(standard_deviation_02)

# Tworzenie histogramu
plt.hist(filtered_signal_02, bins=100, density=True, color='b', alpha=0.7)

title = f'Filtr Fc=0,2 F=2000'
plt.title(title)
plt.xlabel('Wartość')
plt.ylabel('Prawdopodobieństwo')
plt.legend(['Histogram'])
plt.grid(True)
plt.show()

#obliczanie dystrybuanty Filtr Fc=0,2 F=2000
sorted_data_02 = np.sort(filtered_signal_02)
y = np.arange(len(sorted_data_02)) / float(len(sorted_data_02))

# Wykres dystrubuanty  Filtr Fc=0,2 F=2000
plt.figure(figsize=(10, 6))
plt.plot(sorted_data_02, y, color='black', marker='.', linestyle='none')
plt.hist(filtered_signal_02, bins=100, density=True,cumulative=True, color='b', alpha=0.7)
plt.title('Dystrybuanta filtra Fc=0,2 F=2000')
plt.xlabel('Wartość')
plt.ylabel('Prawdopodobieństwo')
plt.grid(True)
plt.yticks(np.arange(0,1.1,0.1))
plt.show()


# Obliczanie kowariancji dla wszystkich par punktów danych
covariance_values_02 = []
for lag in range(0, 100):  # zakres opóźnień (lag)
    shifted_data = np.roll(filtered_signal_02, lag)
    covariance_02 = np.mean((filtered_signal_02 - np.mean(filtered_signal_02)) * (shifted_data - np.mean(shifted_data)))
    covariance_values_02.append(covariance_02)

# Tworzenie wykresu funkcji kowariancyjnej
plt.figure(figsize=(10, 5))
plt.plot(czas, covariance_values_02, color='blue')
plt.title('Funkcja kowariancyjna dla Filtra 02')
plt.xlabel('Czas (s)')
plt.ylabel('Wartość kowariancji')
plt.grid(True)
plt.show()

# Obliczanie pierwszego zera funkcji kowariancji po filtracji
first_zero_index_02 = next((i for i, val in enumerate(covariance_values_02) if val <= 0), None)

if first_zero_index_02 is not None:
    first_zero_delay_02 = first_zero_index_02 * (1 / sampling_rate)  # Opóźnienie dla pierwszego zera, przeliczone na sekundy
    print("Pierwsze zero funkcji kowariancji po filtracji 02:", first_zero_delay_02, "s")
else:
    print("Brak zer w funkcji kowariancji po filtracji 02.")


# zadanie 3 i 4 dla Fc=0,4 - to takie dodatkowe dla pokazania że robi się dołek większy
#można też dać dla 0,5 to powinno być barzdiej widaoczne

# Parametry filtru
Fc = 0.4  # Częstotliwość odcięcia
F = 4000  # Częstotliwość próbkowania sygnału wąskopasmowego 10k*0,4
M = 31  # Długość odpowiedzi impulsowej filtra
taps = firwin(M, Fc)

# Filtracja
filtered_signal_04 = lfilter(taps, 1.0, white_noise)

# Wykres sygnału wąskopasmowego
plt.plot(t, filtered_signal_04, 'r-', linewidth=2, label='Sygnał dolnopasmowy')
plt.title(f'Fc = {Fc}, F = {F} Hz')
plt.xlabel('Czas [s]')
plt.ylabel('Amplituda')
plt.legend()
plt.show()

# Wykres sygnału wąskopasmowego  i gaussowski
plt.plot(t, white_noise, 'b-', label='Sygnał oryginalny')
plt.plot(t, filtered_signal_04, 'r-', linewidth=2, label='Sygnał dolnopasmowy')
plt.title(f'Fc = {Fc}, F = {F} Hz')
plt.xlabel('Czas [s]')
plt.ylabel('Amplituda')
plt.legend()
plt.show()

# Obliczanie wartości oczekiwanej i wariancji dla filtru 04
expected_value_04 = np.mean(filtered_signal_04)
variance_04 = np.var(filtered_signal_04)
print("")
print("wartość oczekiwana filtru 04:")
print(expected_value_04)
print("")
print("wariancja filtru 04:")
print(variance_04)

# odchylenie standarowe ze wzoru co podał na tablicy
standard_deviation_04 = np.sqrt(variance_04)
print("")
print("Odchylenie standardowe filtru 04:")
print(standard_deviation_04)

# Tworzenie histogramu
plt.hist(filtered_signal_04, bins=100, density=True, color='b', alpha=0.7)

# Dodanie wykresu gęstości prawdopodobieństwa
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 100)
# p = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-((x - mean) ** 2) / (2 * std ** 2))
# plt.plot(x, p, 'k', linewidth=2)

title = f'Filtr Fc=0,4 F=500'
plt.title(title)
plt.xlabel('Wartość')
plt.ylabel('Prawdopodobieństwo')
plt.legend(['Histogram'])
plt.grid(True)
plt.show()

#obliczanie dystrybuanty Filtr Fc=0,4 F=500
sorted_data_04 = np.sort(filtered_signal_04)
y = np.arange(len(sorted_data_04)) / float(len(sorted_data_04))

# Wykres dystrubuanty  Filtr Fc=0,4 F=500
plt.figure(figsize=(10, 6))
plt.plot(sorted_data_04, y, color='black', marker='.', linestyle='none')
plt.hist(filtered_signal_04, bins=100, density=True,cumulative=True, color='b', alpha=0.7)
plt.title('Dystrybuanta filtra Fc=0,4 F=500')
plt.xlabel('Wartość')
plt.ylabel('Prawdopodobieństwo')
plt.grid(True)
plt.yticks(np.arange(0,1.1,0.1))
plt.show()


# Obliczanie kowariancji dla wszystkich par punktów danych
covariance_values_04 = []
for lag in range(0, 100):  # zakres opóźnień (lag)
    shifted_data = np.roll(filtered_signal_04, lag)
    covariance_04 = np.mean((filtered_signal_04 - np.mean(filtered_signal_04)) * (shifted_data - np.mean(shifted_data)))
    covariance_values_04.append(covariance_04)

# Tworzenie wykresu funkcji kowariancyjnej
plt.figure(figsize=(10, 5))
plt.plot(czas, covariance_values_04, color='blue')
plt.title('Funkcja kowariancyjna dla Filtra 04')
plt.xlabel('Czas (s)')
plt.ylabel('Wartość kowariancji')
plt.grid(True)
plt.show()