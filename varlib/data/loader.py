"""
varlib.data.loader
==================
Adquisición y preprocesamiento de datos de precios financieros.

Clase principal
---------------
DataLoader
    Descarga series históricas de precios con yfinance, calcula
    rendimientos logarítmicos o simples, y proporciona una partición
    train/test lista para los modelos VaR.

Convención de rendimientos
--------------------------
Todos los modelos de varlib están pensados para rendimientos LOGARÍTMICOS,
que son aditivos en el tiempo y coherentes con el escalado a horizonte h
(regla de la raíz del tiempo). Usa get_log_returns() (lo habitual).
get_simple_returns() existe por completitud, pero avisa con un UserWarning
porque puede producir resultados incoherentes con los modelos.

Ejemplo de uso
--------------
from varlib.data.loader import DataLoader
loader = DataLoader("AAPL", start="2020-01-01", end="2024-12-31")
loader.download()
returns = loader.get_log_returns()
train, test = loader.split(warmup=250)
"""

import warnings

import numpy as np
import pandas as pd
import yfinance as yf


class DataLoader:
    """
    Descarga y preprocesa series históricas de precios de un activo.

    Parameters
    ----------
    ticker : str
        Símbolo del activo (p.ej. "AAPL", "^IBEX", "BTC-USD", "EURUSD=X").
    start : str
        Fecha de inicio en formato "YYYY-MM-DD".
    end : str
        Fecha de fin en formato "YYYY-MM-DD" (no incluida por yfinance).
    interval : str, optional
        Frecuencia de los datos. Valores admitidos por yfinance:
        "1d" (defecto), "1wk", "1mo", etc.
    price_col : str, optional
        Columna de precio a utilizar. Por defecto "Close".

    Attributes
    ----------
    prices_ : pd.Series
        Serie de precios descargada tras llamar a download.
    _returns_stored : dict
        Caché interna de rendimientos ya calculados.

    Raises
    ------
    RuntimeError
        Si se intenta acceder a rendimientos o precios antes de llamar
        a download.
    ValueError
        Si los parámetros de entrada no son válidos.
    """

    _VALID_INTERVALS = ("1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo")

    def __init__(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str = "1d",
        price_col: str = "Close",
    ) -> None:
        if not isinstance(ticker, str):
            raise ValueError("[ticker] debe ser una cadena.")
        if " " in ticker or "," in ticker:
            raise ValueError("[ticker] debe ser un único activo sin espacios ni comas.")
        if interval not in self._VALID_INTERVALS:
            raise ValueError(
                f"[interval] no reconocido: '{interval}'. "
                f"Opciones válidas: {self._VALID_INTERVALS}"
            )

        self.ticker: str = ticker.upper()
        self.start: str = start
        self.end: str = end
        self.interval: str = interval
        self.price_col: str = price_col

        self.prices_: pd.Series | None = None
        self._returns_stored: dict[str, pd.Series] = {}

        # Métricas de completitud de la descarga (se rellenan en download()).
        self.n_raw_: int | None = None   # filas que entregó yfinance
        self.n_nan_: int | None = None   # NAs eliminados en la columna de precio

    # ── Descarga ───────────────────────────────────────────────────────────────

    def download(self) -> "DataLoader":
        """
        Descarga los precios históricos con yfinance.

        Descarga el OHLCV del activo configurado, extrae la columna de
        precio indicada y elimina valores ausentes.

        Returns
        -------
        DataLoader
            La propia instancia para permitir encadenamiento de métodos.

        Raises
        ------
        ValueError
            Si yfinance no devuelve datos para el ticker o el rango indicado.

        Examples
        --------
        loader = DataLoader("MSFT", "2022-01-01", "2024-01-01")
        loader.download()
        loader.prices_.head()
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_data: pd.DataFrame = yf.download(
                self.ticker,
                start=self.start,
                end=self.end,
                interval=self.interval,
                auto_adjust=True,
                progress=False,
            )

        if raw_data.empty:
            raise ValueError(
                f"yfinance no devolvió datos para '{self.ticker}' "
                f"en el rango [{self.start}, {self.end}] con intervalo '{self.interval}'."
            )

        # yfinance puede devolver MultiIndex si se piden varios tickers,
        # aquí siempre es un único ticker.
        if isinstance(raw_data.columns, pd.MultiIndex):
            raw_data.columns = raw_data.columns.droplevel(1)

        if self.price_col not in raw_data.columns:
            available = list(raw_data.columns)
            raise ValueError(
                f"Columna '{self.price_col}' no encontrada. "
                f"Columnas disponibles: {available}"
            )

        price_series = raw_data[self.price_col]

        # Métricas de completitud: se calculan ANTES de limpiar, porque tras
        # el dropna() se pierde la información de cuántos NAs entregó yfinance.
        self.n_raw_ = len(price_series)
        self.n_nan_ = int(price_series.isna().sum())

        self.prices_ = price_series.dropna()
        self.prices_.name = self.ticker
        self._returns_stored.clear()   # vacía los rendimientos guardados al re-descargar

        if self.n_nan_ > 0:
            warnings.warn(
                f"yfinance devolvió {self.n_nan_} NA(s) en '{self.price_col}' para "
                f"'{self.ticker}', eliminados ({len(self.prices_)}/{self.n_raw_} válidos).",
                stacklevel=2,
            )
        return self

    # ── Rendimientos ───────────────────────────────────────────────────────────

    def _check_downloaded(self) -> None:
        """
        Lanza RuntimeError si no se han descargado precios.
        """
        if self.prices_ is None:
            raise RuntimeError(
                "Primero debes llamar a .download() para obtener los precios."
            )

    def get_log_returns(self) -> pd.Series:
        """
        Calcula los rendimientos logarítmicos diarios.

        Fórmula: r[t] = ln(P[t] / P[t-1])

        Returns
        -------
        pd.Series
            Serie de rendimientos log con el mismo índice de fechas que
            prices_, salvo la primera fecha, para la que no existe rendimiento.

        Examples
        --------
        returns = loader.get_log_returns()
        returns.describe()
        """
        self._check_downloaded()

        if "log" not in self._returns_stored:
            log_ret = np.log(self.prices_ / self.prices_.shift(1)).dropna()
            log_ret.name = f"{self.ticker}_log_returns"
            self._returns_stored["log"] = log_ret

        return self._returns_stored["log"]

    def get_simple_returns(self) -> pd.Series:
        """
        Calcula los rendimientos simples (aritméticos) diarios.

        Fórmula: r[t] = (P[t] - P[t-1]) / P[t-1]

        Aviso
        -----
        Los modelos de varlib están pensados para rendimientos LOGARÍTMICOS
        (aditivos en el tiempo, coherentes con el escalado a horizonte). Usar
        rendimientos simples puede dar resultados inconsistentes, por lo que se
        emite un UserWarning. Usa get_log_returns() salvo que sepas lo que haces.

        Returns
        -------
        pd.Series
            Serie de rendimientos simples con índice de fechas.

        Examples
        --------
        returns = loader.get_simple_returns()
        """
        self._check_downloaded()

        if "simple" not in self._returns_stored:
            warnings.warn(
                "Los modelos de varlib esperan rendimientos LOGARÍTMICOS "
                "(aditivos en el tiempo). Usar rendimientos simples puede dar "
                "resultados incoherentes con el escalado a horizonte. Usa "
                "get_log_returns() salvo que sepas lo que haces.",
                UserWarning,
                stacklevel=2,
            )
            simple_ret = self.prices_.pct_change().dropna()
            simple_ret.name = f"{self.ticker}_simple_returns"
            self._returns_stored["simple"] = simple_ret

        return self._returns_stored["simple"]

    # ── Partición train / test ─────────────────────────────────────────────────

    def split(
        self,
        warmup: int = 250,
        return_type: str = "log",
    ) -> tuple[pd.Series, pd.Series]:
        """
        Divide los rendimientos en warm-up y evaluación.

        La partición es temporal y por número de días, no por porcentaje: los
        primeros `warmup` días sirven de historia inicial para arrancar el
        modelo y el resto se evalúa out-of-sample. Es la unidad natural en
        backtesting de VaR (cuánta historia necesita el modelo), y deja el
        periodo de evaluación lo más amplio posible. Por defecto 250 días
        (~1 año bursátil), que coincide con la ventana deslizante habitual.

        Parameters
        ----------
        warmup : int, optional
            Número de días iniciales de warm-up (entrenamiento). Debe estar
            en (0, n_returns). Por defecto 250.
        return_type : {"log", "simple"}, optional
            Tipo de rendimientos sobre los que realizar la partición.
            Por defecto "log".

        Returns
        -------
        train : pd.Series
            Rendimientos de warm-up (primeros `warmup` días).
        test : pd.Series
            Rendimientos de evaluación (el resto).

        Raises
        ------
        ValueError
            Si warmup no es un entero en (0, n_returns).
        ValueError
            Si return_type no es "log" ni "simple".

        Examples
        --------
        train, test = loader.split(warmup=250)
        print(len(train), len(test))
        """
        if return_type == "log":
            returns = self.get_log_returns()
        elif return_type == "simple":
            returns = self.get_simple_returns()
        else:
            raise ValueError("[return_type] debe ser 'log' o 'simple'.")

        if not isinstance(warmup, int) or not 0 < warmup < len(returns):
            raise ValueError(
                f"[warmup] debe ser un entero en (0, {len(returns)}). "
                f"Recibido: '{warmup}'"
            )

        return returns.iloc[:warmup], returns.iloc[warmup:]

    # ── Información ───────────────────────────────────────────────────────────

    def summary(self, return_type: str = "log") -> dict:
        """
        Devuelve un resumen estadístico básico del activo descargado.

        Returns
        -------
        dict
            Diccionario con claves: ticker, start, end, interval,
            n_raw, n_nan_dropped, pct_complete, n_prices, n_returns,
            mean_return, std_return, min_return, max_return,
            skewness, kurtosis.

            n_raw, n_nan_dropped y pct_complete describen la calidad de la
            descarga: cuántas filas entregó yfinance, cuántos NAs se
            eliminaron y qué fracción resultó utilizable.

        Raises
        ------
        RuntimeError
            Si no se han descargado datos aún.

        Examples
        --------
        info = loader.summary()
        print(info["mean_return"])
        """
        self._check_downloaded()
        if return_type == "log":
            returns = self.get_log_returns()
        elif return_type == "simple":
            returns = self.get_simple_returns()
        else:
            raise ValueError("[return_type] debe ser 'log' o 'simple'.")

        pct_complete = len(self.prices_) / self.n_raw_ if self.n_raw_ else float("nan")

        return {
            "ticker":        self.ticker,
            "start":         str(self.prices_.index[0].date()),
            "end":           str(self.prices_.index[-1].date()),
            "interval":      self.interval,
            "n_raw":         self.n_raw_,        # filas que entregó yfinance
            "n_nan_dropped": self.n_nan_,        # NAs eliminados de la columna de precio
            "pct_complete":  pct_complete,       # precios válidos / filas descargadas
            "n_prices":      len(self.prices_),
            "n_returns":     len(returns),
            "mean_return":   float(returns.mean()),
            "std_return":    float(returns.std(ddof=1)),
            "min_return":    float(returns.min()),
            "max_return":    float(returns.max()),
            "skewness":      float(returns.skew()),
            "kurtosis":      float(returns.kurt()),   # exceso de curtosis
        }

    # ── Representación ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        status = "descargado" if self.prices_ is not None else "sin descargar"
        return (
            f"DataLoader(ticker='{self.ticker}', "
            f"start='{self.start}', end='{self.end}', "
            f"interval='{self.interval}', status={status})"
        )
