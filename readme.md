# Rankability

This repository implements the methods introduced in our paper to quantify the confidence and stability of data orderings.

## Paper Citation & Link

If you use this code or method, please cite our paper:
> **On the fuzzy entropy and the rankability of data**  
> *IEEE Open Journal of the Computer Society*, 2025.  
> [Read the full paper on IEEE Xplore](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11288023)

```bibtex
@article{kong2025fuzzy,
  title={On the fuzzy entropy and the rankability of data},
  author={Kong, Lingping and Vel{\'a}squez, Juan D and Pant, Millie and Pan, Jeng-Shyang and Sn{\'a}{\v{s}}el, V{\'a}clav},
  journal={IEEE Open Journal of the Computer Society},
  year={2025},
  publisher={IEEE}
}
```

### Overview
Rankability fundamentally differs from ranking: while ranking generates an order that shifts with feature importance, rankability measures the confidence or stability of that ordering. This paper addresses the computational limitations of existing methods by proposing a novel, fast, and interpretable rankability measure based on **entropy and variance**. We evaluate this measure on football tournament datasets, demonstrating a strong correlation with well-established ranking systems like the Elo rating, making it highly practical for environments lacking ground-truth consensus.

---

## Source Data

The tournament and game datasets used in this project are publicly available. Please download them directly from the original repository:
*   **Dataset Link:** [specR GitHub Repository (by trcameron)](https://github.com/trcameron/specR)

---

## License

This project is licensed under the Creative Commons Attribution 4.0 International License - see the [LICENSE](LICENSE) file for details.
