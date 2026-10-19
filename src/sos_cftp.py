from pathlib import Path
from time import monotonic

import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs_mixte_moinsN_plusN"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def total_energy(grid):
    return int(
        np.abs(grid[1:, :] - grid[:-1, :]).sum()
        + np.abs(grid[:, 1:] - grid[:, :-1]).sum()
    )


def local_energy(grid, i, j, h_val):
    L = grid.shape[0]

    e = 0

    if i > 0:
        e += abs(h_val - grid[i - 1, j])
    if i < L - 1:
        e += abs(h_val - grid[i + 1, j])
    if j > 0:
        e += abs(h_val - grid[i, j - 1])
    if j < L - 1:
        e += abs(h_val - grid[i, j + 1])

    return e


def apply_step(grid, i, j, delta, u, K, beta):
    h_old = int(grid[i, j])
    h_new = h_old + delta

    if h_new < -K or h_new > K:
        return

    dH = local_energy(grid, i, j, h_new) - local_energy(grid, i, j, h_old)

    if dH <= 0 or u < np.exp(-beta * dH):
        grid[i, j] = h_new


def make_max_mixed(N, K):
    L = 2 * N + 1

    h = np.full((L, L), K, dtype=np.int64)

    # Conditions mixtes sur {-N,...,N}^2 :
    # bas fixe a -N, haut fixe a +N, cotes gauche/droite libres.
    h[0, :] = -N
    h[L - 1, :] = N

    return h


def make_min_mixed(N, K):
    L = 2 * N + 1

    h = np.full((L, L), -K, dtype=np.int64)

    h[0, :] = -N
    h[L - 1, :] = N

    return h


def propp_wilson_mixed(
    N=12,
    beta=2.0,
    seed=0,
    heartbeat_seconds=30.0,
):
    L = 2 * N + 1
    K = N + 1

    rng = np.random.default_rng(seed)
    start = monotonic()

    # Sites libres :
    # i_python = 1,...,L-2 donc i_math = -N+1,...,N-1
    # j_python = 0,...,L-1 donc j_math = -N,...,N
    # Les bords gauche/droite sont libres.
    sites_i = np.repeat(np.arange(1, L - 1, dtype=np.int16), L)
    sites_j = np.tile(np.arange(0, L, dtype=np.int16), L - 2)
    M = len(sites_i)

    print(
        f"[RUNNING] Propp-Wilson mixte beta={beta}, "
        f"N_math={N}, L={L}, K={K}, sites libres={M}, aucune limite",
        flush=True,
    )

    blocks = []
    T = M
    n_steps_total = 0
    n_iter = 0

    while True:
        n_iter += 1

        idx_sites = rng.integers(0, M, size=T, dtype=np.int32)
        deltas = (2 * rng.integers(0, 2, size=T, dtype=np.int8) - 1).astype(np.int8)
        us = rng.random(size=T)

        blocks.append((idx_sites, deltas, us))
        n_steps_total += T

        print(
            f"[RUNNING] iteration {n_iter}: replay de {n_steps_total:,} etapes",
            flush=True,
        )

        h_min = make_min_mixed(N, K)
        h_max = make_max_mixed(N, K)

        replayed = 0
        next_heartbeat = monotonic() + heartbeat_seconds
        next_check = 100_000

        for idx_block, delta_block, u_block in reversed(blocks):
            for idx, delta, u in zip(idx_block, delta_block, u_block):
                i = int(sites_i[idx])
                j = int(sites_j[idx])
                delta = int(delta)
                u = float(u)

                apply_step(h_min, i, j, delta, u, K, beta)
                apply_step(h_max, i, j, delta, u, K, beta)

                replayed += 1

                if replayed >= next_check:
                    now = monotonic()
                    if now >= next_heartbeat:
                        print(
                            f"    toujours en cours: "
                            f"{replayed:,}/{n_steps_total:,} etapes rejouees",
                            flush=True,
                        )
                        next_heartbeat = now + heartbeat_seconds
                    next_check += 100_000

        diff = int(np.max(h_max - h_min))
        coalesced = np.all(h_min == h_max)

        print(
            f"[CHECK] T={n_steps_total:,}  max-min={diff}  "
            f"{'COALESCENCE' if coalesced else ''}",
            flush=True,
        )

        if coalesced:
            elapsed = (monotonic() - start) / 60.0
            print(
                f"[DONE] Coalescence en {n_steps_total:,} etapes "
                f"apres {elapsed:.1f} minutes",
                flush=True,
            )
            return h_min, n_steps_total, elapsed

        T = n_steps_total



# ======================================================================
# AJOUT : dynamique hybride proposee par M. Laslier
# ======================================================================
#
# On conserve EXACTEMENT les mouvements Metropolis locaux ci-dessus.
# A chaque etape :
#   1) on effectue un mouvement Metropolis local ordinaire ;
#   2) avec une petite probabilite p, on choisit une ligne libre
#      uniformement et on effectue k mouvements Metropolis locaux
#      supplementaires sur cette meme ligne.
#
# Par defaut p = 1/M, ou M est le nombre de sites libres :
# cela donne en moyenne un "bombardement" de ligne par balayage
# complet de M mouvements locaux ordinaires.
#
# Tous les aleas sont stockes, exactement comme dans le CFTP original,
# afin que le meme passe soit rejoue lorsque l'horizon est double.
# ======================================================================


def propp_wilson_mixed_with_line_bombardments(
    N=12,
    beta=3.0,
    k=100,
    seed=0,
    heartbeat_seconds=30.0,
    bombardment_probability=None,
):
    L = 2 * N + 1
    K = N + 1

    rng = np.random.default_rng(seed)
    start = monotonic()

    # Sites libres : exactement comme dans propp_wilson_mixed.
    sites_i = np.repeat(np.arange(1, L - 1, dtype=np.int16), L)
    sites_j = np.tile(np.arange(0, L, dtype=np.int16), L - 2)
    M = len(sites_i)

    # Par defaut : un bombardement en moyenne tous les M pas locaux,
    # donc environ un bombardement par balayage complet du systeme.
    if bombardment_probability is None:
        bombardment_probability = 1.0 / M

    p = float(bombardment_probability)
    if not (0.0 <= p <= 1.0):
        raise ValueError("bombardment_probability doit etre dans [0,1].")
    if k < 1:
        raise ValueError("k doit etre >= 1.")

    print(
        f"[RUNNING] Propp-Wilson HYBRIDE beta={beta}, "
        f"N_math={N}, L={L}, K={K}, sites libres={M}",
        flush=True,
    )
    print(
        f"[RUNNING] mouvements locaux ordinaires + bombardement occasionnel : "
        f"k={k}, p={p:.8f} ~= 1 bombardement tous les {1.0/p if p > 0 else float('inf'):.1f} pas ordinaires",
        flush=True,
    )

    # Chaque element de blocks contient :
    # - les T mouvements locaux ordinaires ;
    # - les instants parmi ces T ou un bombardement a lieu ;
    # - pour chacun de ces bombardements :
    #       la ligne choisie et les k aleas locaux sur cette ligne.
    blocks = []

    T = M
    n_steps_total = 0               # nombre de pas locaux ORDINAIRES dans l'horizon
    n_bombardments_total = 0        # nombre de bombardements dans l'horizon
    n_iter = 0

    while True:
        n_iter += 1

        # --------------------------------------------------------------
        # Partie Metropolis locale : IDENTIQUE au code original.
        # --------------------------------------------------------------
        idx_sites = rng.integers(0, M, size=T, dtype=np.int32)
        deltas = (2 * rng.integers(0, 2, size=T, dtype=np.int8) - 1).astype(np.int8)
        us = rng.random(size=T)

        # --------------------------------------------------------------
        # AJOUT : on decide, apres chacun de ces pas ordinaires,
        # s'il y a un bombardement de ligne.
        # --------------------------------------------------------------
        bombard_mask = rng.random(size=T) < p
        bombard_times = np.flatnonzero(bombard_mask).astype(np.int32)
        n_bomb = len(bombard_times)

        # Une ligne libre uniforme pour chaque bombardement.
        bombard_lines = rng.integers(
            1, L - 1, size=n_bomb, dtype=np.int16
        )

        # Pour chaque bombardement, k mises a jour locales sur cette ligne.
        # La colonne, le signe et l'uniforme sont tous stockes.
        bombard_js = rng.integers(
            0, L, size=(n_bomb, k), dtype=np.int16
        )
        bombard_deltas = (
            2 * rng.integers(0, 2, size=(n_bomb, k), dtype=np.int8) - 1
        ).astype(np.int8)
        bombard_us = rng.random(size=(n_bomb, k))

        blocks.append(
            (
                idx_sites,
                deltas,
                us,
                bombard_times,
                bombard_lines,
                bombard_js,
                bombard_deltas,
                bombard_us,
            )
        )

        n_steps_total += T
        n_bombardments_total += n_bomb

        # Cout de l'histoire finale, exprime en propositions Metropolis locales.
        n_local_proposals = n_steps_total + k * n_bombardments_total

        print(
            f"[RUNNING] iteration {n_iter}: "
            f"horizon={n_steps_total:,} pas ordinaires, "
            f"bombardements={n_bombardments_total:,}, "
            f"cout local={n_local_proposals:,}",
            flush=True,
        )

        h_min = make_min_mixed(N, K)
        h_max = make_max_mixed(N, K)

        replayed = 0
        replayed_local_proposals = 0
        next_heartbeat = monotonic() + heartbeat_seconds
        next_check = 100_000

        # Rejeu du passe du plus ancien au plus recent,
        # exactement selon le principe du code original.
        for (
            idx_block,
            delta_block,
            u_block,
            bombard_times_block,
            bombard_lines_block,
            bombard_js_block,
            bombard_deltas_block,
            bombard_us_block,
        ) in reversed(blocks):

            bomb_pos = 0
            n_bomb_block = len(bombard_times_block)

            for t, (idx, delta, u) in enumerate(
                zip(idx_block, delta_block, u_block)
            ):
                # ------------------------------------------------------
                # 1) mouvement Metropolis local ORDINAIRE
                # ------------------------------------------------------
                i = int(sites_i[idx])
                j = int(sites_j[idx])
                delta = int(delta)
                u = float(u)

                apply_step(h_min, i, j, delta, u, K, beta)
                apply_step(h_max, i, j, delta, u, K, beta)

                replayed += 1
                replayed_local_proposals += 1

                # ------------------------------------------------------
                # 2) PARFOIS : bombardement d'UNE ligne
                # ------------------------------------------------------
                if (
                    bomb_pos < n_bomb_block
                    and t == int(bombard_times_block[bomb_pos])
                ):
                    i_line = int(bombard_lines_block[bomb_pos])

                    for j_line, delta_line, u_line in zip(
                        bombard_js_block[bomb_pos],
                        bombard_deltas_block[bomb_pos],
                        bombard_us_block[bomb_pos],
                    ):
                        j_line = int(j_line)
                        delta_line = int(delta_line)
                        u_line = float(u_line)

                        # Memes aleas pour les deux copies du couplage.
                        apply_step(
                            h_min,
                            i_line,
                            j_line,
                            delta_line,
                            u_line,
                            K,
                            beta,
                        )
                        apply_step(
                            h_max,
                            i_line,
                            j_line,
                            delta_line,
                            u_line,
                            K,
                            beta,
                        )

                        replayed_local_proposals += 1

                    bomb_pos += 1

                if replayed >= next_check:
                    now = monotonic()
                    if now >= next_heartbeat:
                        print(
                            f"    toujours en cours: "
                            f"{replayed:,}/{n_steps_total:,} pas ordinaires rejoues, "
                            f"{replayed_local_proposals:,} propositions locales effectuees",
                            flush=True,
                        )
                        next_heartbeat = now + heartbeat_seconds
                    next_check += 100_000

        diff = int(np.max(h_max - h_min))
        coalesced = np.all(h_min == h_max)

        print(
            f"[CHECK] horizon={n_steps_total:,} pas ordinaires  "
            f"bombardements={n_bombardments_total:,}  "
            f"cout local={n_local_proposals:,}  "
            f"max-min={diff}  "
            f"{'COALESCENCE' if coalesced else ''}",
            flush=True,
        )

        if coalesced:
            elapsed = (monotonic() - start) / 60.0
            print(
                f"[DONE] Coalescence : "
                f"{n_steps_total:,} pas ordinaires + "
                f"{n_bombardments_total:,} bombardements de {k} pas, "
                f"soit {n_local_proposals:,} propositions locales "
                f"apres {elapsed:.1f} minutes",
                flush=True,
            )
            return (
                h_min,
                n_steps_total,
                n_bombardments_total,
                n_local_proposals,
                elapsed,
            )

        # Meme doublement de l'horizon que dans le CFTP original.
        T = n_steps_total



def save_outputs(grid, n_steps, elapsed, N, beta, seed):
    L = 2 * N + 1
    mid = N

    prefix = f"mixte_moinsN_plusN_beta_{beta:g}_N_{N}"

    matrix_txt = OUTPUT_DIR / f"{prefix}_matrice.txt"
    matrix_csv = OUTPUT_DIR / f"{prefix}_matrice.csv"
    matrix_excel = OUTPUT_DIR / f"{prefix}_matrice_excel_fr.csv"
    summary = OUTPUT_DIR / f"{prefix}_resume.txt"

    np.savetxt(matrix_txt, grid, fmt="%4d")
    np.savetxt(matrix_csv, grid, fmt="%d", delimiter=",")
    np.savetxt(matrix_excel, grid, fmt="%d", delimiter=";")

    with open(summary, "w", encoding="utf-8") as f:
        f.write(f"N_math = {N}\n")
        f.write(f"L = 2*N+1 = {L}\n")
        f.write(f"beta = {beta}\n")
        f.write(f"seed = {seed}\n")
        f.write(f"T coalescence = {n_steps}\n")
        f.write(f"temps minutes = {elapsed:.3f}\n")
        f.write(f"H = {total_energy(grid)}\n")
        f.write(f"h centre = {int(grid[mid, mid])}\n")
        f.write(f"min = {int(grid.min())}\n")
        f.write(f"max = {int(grid.max())}\n")
        f.write(f"moyenne = {float(grid.mean())}\n")

    print(f"[DONE] Matrice TXT : {matrix_txt}", flush=True)
    print(f"[DONE] Matrice CSV : {matrix_csv}", flush=True)
    print(f"[DONE] Resume : {summary}", flush=True)


def plot_outputs(grid, n_steps, N, beta, show=True):
    """
    Cree, enregistre automatiquement et affiche les graphiques.

    Fichiers crees dans OUTPUT_DIR :
      - configuration 2D + profil vertical
      - surface 3D
      - configuration h(i,j)-i
      - configuration centree par la moyenne globale
      - matrice des valeurs
      - un PDF contenant toutes les figures

    Si show=True, toutes les figures sont affichees a la fin.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    L = 2 * N + 1
    coords = np.arange(-N, N + 1)
    X, Y = np.meshgrid(coords, coords)

    prefix = f"mixte_moinsN_plusN_beta_{beta:g}_N_{N}"

    pdf_path = OUTPUT_DIR / f"{prefix}_graphes_complets.pdf"
    fig2d_path = OUTPUT_DIR / f"{prefix}_configuration_2d.png"
    fig3d_path = OUTPUT_DIR / f"{prefix}_configuration_3d.png"
    fig_moins_i_path = OUTPUT_DIR / f"{prefix}_moins_ordonnee.png"
    fig_moins_moy_path = OUTPUT_DIR / f"{prefix}_moins_moyenne_globale.png"
    fig_matrice_path = OUTPUT_DIR / f"{prefix}_matrice_valeurs.png"

    h_minus_i = grid - coords[:, None]
    h_minus_mean = grid - grid.mean()
    profil_vertical = grid.mean(axis=1)

    figures = []

    with PdfPages(pdf_path) as pdf:
        # ==============================================================
        # Figure 1 : configuration 2D + profil vertical
        # ==============================================================
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        vmax = max(abs(int(grid.min())), abs(int(grid.max())))
        im = axes[0].imshow(
            grid,
            origin="lower",
            extent=[-N, N, -N, N],
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        plt.colorbar(im, ax=axes[0], label="h(i,j)")
        axes[0].set_title("Configuration exacte Propp-Wilson")
        axes[0].set_xlabel("j")
        axes[0].set_ylabel("i")

        axes[1].plot(profil_vertical, coords, marker="o", lw=1.4)
        axes[1].axvline(0, color="gray", ls="--", lw=0.8)
        axes[1].set_title("Profil vertical moyen")
        axes[1].set_xlabel("moyenne en j")
        axes[1].set_ylabel("i")
        axes[1].grid(True, alpha=0.25)

        fig.suptitle(
            f"Mixte sur {{-N,...,N}}^2 - N={N}, L={L}, beta={beta}, "
            f"T={n_steps:,}",
            fontsize=11,
        )
        fig.tight_layout()

        fig.savefig(fig2d_path, dpi=150, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        figures.append(fig)

        # ==============================================================
        # Figure 2 : surface 3D de la configuration exacte
        # ==============================================================
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(1, 1, 1, projection="3d")

        surf = ax.plot_surface(
            X,
            Y,
            grid,
            cmap="RdBu_r",
            edgecolor="none",
            linewidth=0,
            antialiased=True,
        )

        ax.set_title("Surface 3D de la configuration exacte")
        ax.set_xlabel("j")
        ax.set_ylabel("i")
        ax.set_zlabel("h(i,j)")
        ax.view_init(elev=28, azim=-135)

        fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.08, label="h(i,j)")

        fig.suptitle(
            f"Propp-Wilson mixte exact - carre {{-N,...,N}}^2 - beta={beta}",
            fontsize=11,
        )
        fig.tight_layout()

        fig.savefig(fig3d_path, dpi=150, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        figures.append(fig)

        # ==============================================================
        # Figure 3 : configuration avec ordonnee soustraite
        # ==============================================================
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        vmax = max(1.0, float(np.max(np.abs(h_minus_i))))
        im = axes[0].imshow(
            h_minus_i,
            origin="lower",
            extent=[-N, N, -N, N],
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        plt.colorbar(im, ax=axes[0], label="h(i,j)-i")
        axes[0].set_title("Configuration h(i,j)-i")
        axes[0].set_xlabel("j")
        axes[0].set_ylabel("i")

        for col in np.linspace(0, L - 1, min(8, L), dtype=int):
            axes[1].plot(
                coords,
                h_minus_i[:, col],
                lw=1.2,
                label=f"j={coords[col]}",
            )
        axes[1].axhline(0, color="gray", ls="--", lw=0.8)
        axes[1].set_title("Trajectoires verticales h(i,j)-i")
        axes[1].set_xlabel("i")
        axes[1].set_ylabel("h(i,j)-i")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(ncol=2, fontsize=8)

        fig.suptitle(
            f"Ordonnee soustraite - N={N}, beta={beta}",
            fontsize=11,
        )
        fig.tight_layout()

        fig.savefig(fig_moins_i_path, dpi=150, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        figures.append(fig)

        # ==============================================================
        # Figure 4 : configuration centree par moyenne globale
        # ==============================================================
        fig, ax = plt.subplots(figsize=(7, 6))

        vmax = max(1.0, float(np.max(np.abs(h_minus_mean))))
        im = ax.imshow(
            h_minus_mean,
            origin="lower",
            extent=[-N, N, -N, N],
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        plt.colorbar(im, ax=ax, label="h(i,j)-moyenne")
        ax.set_title("Configuration moins moyenne globale")
        ax.set_xlabel("j")
        ax.set_ylabel("i")

        fig.tight_layout()

        fig.savefig(fig_moins_moy_path, dpi=150, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        figures.append(fig)

        # ==============================================================
        # Figure 5 : matrice des valeurs
        # ==============================================================
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.axis("off")
        ax.set_title("Matrice des valeurs h(i,j)")

        table = ax.table(
            cellText=[[f"{int(x):d}" for x in row] for row in grid],
            rowLabels=[str(i) for i in coords],
            colLabels=[str(j) for j in coords],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(5 if L > 25 else 6)
        table.scale(1.0, 1.1)

        fig.tight_layout()

        fig.savefig(fig_matrice_path, dpi=150, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        figures.append(fig)

    # Tous les fichiers sont deja enregistres AVANT l'affichage.
    print(f"[DONE] Figure 2D              : {fig2d_path}", flush=True)
    print(f"[DONE] Figure 3D              : {fig3d_path}", flush=True)
    print(f"[DONE] Figure h(i,j)-i        : {fig_moins_i_path}", flush=True)
    print(f"[DONE] Figure centree         : {fig_moins_moy_path}", flush=True)
    print(f"[DONE] Matrice en image       : {fig_matrice_path}", flush=True)
    print(f"[DONE] PDF complet            : {pdf_path}", flush=True)

    if show:
        print("[DISPLAY] Affichage des graphiques...", flush=True)
        plt.show()

    # On ferme les figures seulement apres l'affichage.
    for fig in figures:
        plt.close(fig)



if __name__ == "__main__":
    # ================================================================
    # EXPERIENCE DEMANDEE
    # ================================================================
    N = 12
    beta = 4.0
    seed = 0

    # CHOISIS ICI : 10, 100 ou 1000.
    k = 100

    # None = p = 1/M.
    # Pour N=12, M=23*25=575 :
    # en moyenne un bombardement tous les 575 pas Metropolis ordinaires.
    bombardment_probability = None

    (
        grid,
        n_steps,
        n_bombardments,
        n_local_proposals,
        elapsed,
    ) = propp_wilson_mixed_with_line_bombardments(
        N=N,
        beta=beta,
        k=k,
        seed=seed,
        heartbeat_seconds=30.0,
        bombardment_probability=bombardment_probability,
    )

    print()
    print("=" * 72)
    print("RESULTAT FINAL")
    print("=" * 72)
    print(f"N                         = {N}")
    print(f"beta                      = {beta}")
    print(f"k                         = {k}")
    print(f"pas Metropolis ordinaires = {n_steps:,}")
    print(f"bombardements de ligne    = {n_bombardments:,}")
    print(f"propositions locales      = {n_local_proposals:,}")
    print(f"temps                      = {elapsed:.3f} min")

    # ================================================================
    # SAUVEGARDE ET AFFICHAGE AUTOMATIQUES
    # ================================================================
    print()
    print("=" * 72)
    print("SAUVEGARDE DES RESULTATS ET CREATION DES GRAPHIQUES")
    print("=" * 72)

    # Sauvegarde de la matrice et du resume texte.
    save_outputs(
        grid=grid,
        n_steps=n_steps,
        elapsed=elapsed,
        N=N,
        beta=beta,
        seed=seed,
    )

    # Enregistre tous les PNG + le PDF complet, puis affiche les figures.
    plot_outputs(
        grid=grid,
        n_steps=n_steps,
        N=N,
        beta=beta,
        show=True,
    )

    print()
    print(f"[DONE] Tous les resultats sont dans : {OUTPUT_DIR}", flush=True)