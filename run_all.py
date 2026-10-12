#!/usr/bin/env python3
"""Roda os 30 algoritmos de IA implementados em numpy."""
from g1_supervisionado import rodar as r1
from g2_boosting import rodar as r2
from g3_nao_supervisionado import rodar as r3
from g4_anomalias import rodar as r4
from g5_reforco import rodar as r5
from g6_redes import rodar as r6


if __name__ == "__main__":
    print("30 Algoritmos de IA — numpy (sem sklearn/xgboost)")
    r1()
    r2()
    r3()
    r4()
    r5()
    r6()
    print("\npronto.")
