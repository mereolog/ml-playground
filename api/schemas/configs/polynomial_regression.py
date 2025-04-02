@dataclass
class PolynomialRegressionParams(SupervisedAlgorithmsParams):
    """
    Ta konfiguracja parametrów modelu definiuje typy i możliwe domyślne argumenty
    używane przez model Regresji Wielomianowej.

    Regresja wielomianowa zazwyczaj przekształca cechy wejściowe na cechy wielomianowe,
    a następnie stosuje regresję liniową. Dlatego wiele parametrów jest podobnych
    do regresji liniowej, ale z dodatkiem stopnia wielomianu.

    Attributes:
        degree: Stopień wielomianu używany do transformacji cech (domyślnie: 2).
        include_bias: Czy dołączać kolumnę biasu (jedynki) podczas transformacji cech (domyślnie: True).
                      Często obsługiwane przez bazowy model regresji liniowej.

        # Parametry odziedziczone lub zaadaptowane z regresji liniowej,
        # stosowane do modelu liniowego na przekształconych cechach:
        learning_rate: Rozmiar kroku dla optymalizacji gradientowej (jeśli używana) (domyślnie: 0.01).
        epochs: Liczba epok dla optymalizacji gradientowej (jeśli używana) (domyślnie: 100).
        batch_size: Rozmiar batcha dla optymalizacji gradientowej (jeśli używana) (domyślnie: None).

        loss: Funkcja straty używana przez bazową regresję liniową ('mse', 'mae') (domyślnie: 'mse').

        reg_type: Typ regularyzacji ('l1', 'l2', 'elasticnet') stosowany do regresji liniowej (domyślnie: None).
        reg_strenght: Siła regularyzacji (lambda/alpha) (domyślnie: 0.01).
        mixing_ratio: Parametr mieszania dla ElasticNet (domyślnie: 0.5).
    """

    degree: int = 2
    include_bias: bool = True

    # -- Parametry bazowej regresji liniowej (jeśli implementacja ich używa) --
    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: Optional[int] = None

    # -- Konfiguracja funkcji straty --
    loss: LossType = "mse" # MSE jest standardem dla regresji

    # -- Konfiguracja regularyzacji --
    reg_type: RegType = None
    reg_strenght: float = 0.01
    mixing_ratio: float = 0.5

    def __post_init__(self):
        """Walidacja parametrów po inicjalizacji."""
        super().__post_init__() # Wywołaj __post_init__ klasy bazowej
        if self.degree < 1:
            raise ValueError("degree musi być liczbą całkowitą >= 1")
        if self.learning_rate <= 0:
             raise ValueError("learning_rate musi być większe od 0")
        if self.epochs <= 0:
             raise ValueError("epochs musi być dodatnią liczbą całkowitą")
        if self.batch_size is not None and self.batch_size <= 0:
             raise ValueError("batch_size musi być dodatnią liczbą całkowitą lub None")
        if self.loss not in ["mse", "mae"]:
             raise ValueError(f"Nieprawidłowa funkcja straty dla regresji: {self.loss}. Dozwolone: 'mse', 'mae'.")
        if self.reg_type is not None and self.reg_strenght <= 0:
            raise ValueError("reg_strenght musi być większe od 0, gdy używana jest regularyzacja")
        if self.reg_type == "elasticnet":
            if not (0.0 <= self.mixing_ratio <= 1.0):
                raise ValueError("mixing_ratio musi być pomiędzy 0 a 1 dla ElasticNet")
        elif self.reg_type is not None and self.mixing_ratio != 0.5:
             print(f"Ostrzeżenie: parametr 'mixing_ratio' ({self.mixing_ratio}) jest ustawiony, "
                   f"ale ma znaczenie tylko gdy reg_type='elasticnet'. Aktualny typ: {self.reg_type}",
                   file=sys.stderr)