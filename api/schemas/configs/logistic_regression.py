@dataclass
class LogisticRegressionParams(SupervisedAlgorithmsParams):
    """
    Ta konfiguracja parametrów modelu definiuje typy i możliwe domyślne argumenty
    używane przez model Regresji Logistycznej.

    Attributes:
        learning_rate: Rozmiar kroku dla optymalizacji gradientowej (domyślnie: 0.01).
        epochs: Liczba pełnych przejść przez zbiór danych treningowych (domyślnie: 100).
        batch_size: Liczba próbek na aktualizację gradientu, None oznacza cały zbiór (full batch) (domyślnie: None).
        threshold: Próg decyzyjny do konwersji prawdopodobieństw na etykiety klas (domyślnie: 0.5).

        reg_type: Typ regularyzacji ('l1', 'l2', 'elasticnet') (domyślnie: None).
        reg_strenght: Siła (lambda/alpha) regularyzacji. Musi być > 0, aby miała efekt (domyślnie: 0.01).
        mixing_ratio: Parametr mieszania dla regularyzacji ElasticNet. Musi być 0 <= mixing_ratio <= 1.
                      Wartość 0.0 odpowiada tylko L2, a 1.0 tylko L1.
                      Używane tylko, gdy reg_type='elasticnet' (domyślnie: 0.5).
        # Uwaga: Parametr 'loss' jest zazwyczaj stały dla regresji logistycznej (np. log loss / binary cross-entropy)
        # i może nie być konfigurowalny w taki sam sposób jak w regresji liniowej.
        # Dla jasności, pominięto go tutaj, zakładając, że implementacja modelu użyje odpowiedniej funkcji straty.
    """

    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: Optional[int] = None
    threshold: float = 0.5

    # -- Konfiguracja regularyzacji --
    reg_type: RegType = None
    reg_strenght: float = 0.01
    mixing_ratio: float = 0.5  # domyślny współczynnik mieszania dla ElasticNet

    def __post_init__(self):
        """Walidacja parametrów po inicjalizacji."""
        super().__post_init__() # Wywołaj __post_init__ klasy bazowej
        if self.learning_rate <= 0:
            raise ValueError("learning_rate musi być większe od 0")
        if self.epochs <= 0:
            raise ValueError("epochs musi być dodatnią liczbą całkowitą")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size musi być dodatnią liczbą całkowitą lub None")
        if not (0.0 < self.threshold < 1.0):
             raise ValueError("threshold musi być wartością pomiędzy 0 a 1 (wyłącznie)")
        if self.reg_type is not None and self.reg_strenght <= 0:
            raise ValueError("reg_strenght musi być większe od 0, gdy używana jest regularyzacja")
        if self.reg_type == "elasticnet":
            if not (0.0 <= self.mixing_ratio <= 1.0):
                raise ValueError("mixing_ratio musi być pomiędzy 0 a 1 dla ElasticNet")
        elif self.reg_type is not None and self.mixing_ratio != 0.5:
            # Ostrzeżenie lub błąd, jeśli mixing_ratio jest ustawione, ale nie używamy ElasticNet
             print(f"Ostrzeżenie: parametr 'mixing_ratio' ({self.mixing_ratio}) jest ustawiony, "
                   f"ale ma znaczenie tylko gdy reg_type='elasticnet'. Aktualny typ: {self.reg_type}",
                   file=sys.stderr)