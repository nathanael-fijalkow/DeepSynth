---
title: 'DeepSynth: Scaling Neural Program Synthesis with Distribution-based Search'
tags:
  - Python
  - program synthesis
  - enumeration
  - scalable
authors:
  - name: Nathanaël Fijalkow
    orcid: 0000-0002-6576-4680  
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Guillaume Lagarde
    affiliation: 1
  - name: Théo Matricon
    affiliation: 1
    orcid: 0000-0002-5043-3221 
  - name: Kevin Ellis
    affiliation: 3
  - name: Pierre Ohlmann
    affiliation: 4
    orcid: 0000-0002-4685-5253
  - name: Akarsh Potta
    affiliation: 5
affiliations:
  - name: CNRS, LaBRI and Université de Bordeaux, France
    index: 1
  - name: The Alan Turing Institute of data science, United Kingdom
    index: 2
  - name: Cornell University, United States
    index: 3
  - name: University of Paris, France
    index: 4
  - name: Indidan Institute of Technology, Bombay, India 
    index: 5
date: 13 December 2021
bibliography: paper.bib
---

# Summary

Writing software is tedious, error-prone, and accessible only to a small share of the population – yet coding grows increasingly important as the digital world plays larger and larger roles in peoples’ lives.
One way to specify such programs is from pairs of input-output examples.
`DeepSynth` is such a framework. Contrary to other approaches, `DeepSynth` leverages the speed of neural networks with the theoretical guarantees of symbolic methods to generate programs, combining the benefits of both worlds to make program synthesis scalable with guarantees. This framework is the authors' implementation of @Fijalkow:2021.

# Statement of need

A common problem in program synthesis is to generate programs based on examples (PBE).  This automated synthesis shines when applied on domain specific languages (DSL). The DSL produced by the user can then be processed to generate programs however doing so is complex and the search for likely programs that fit the given examples is hard.
`DeepSynth` proposes an automated pipeline for engineers, researchers and students to generate programs based on pairs of input-output examples.
\autoref{fig:description} illustrates the machine learning pipeline for program synthesis on a toy DSL describing integer list manipulating program. The user only needs to specify the setting and the input output examples. The setting represents the semantic and syntaxic constraints describing the DSL, this can be implemented by an expert. The pairs of input-output examples can then be given the end user.
This software is the implementation associated to @Fijalkow:2021.
This framework leverages @pytorch to produce a neural network to guide the search based on examples towards likely programs.
Then the enumeration of likely programs can be done with algorithms such as `HeapSearch` that are proved optimal yet are faster than previous optimal enumeration algorithms such as $A^*$.
Furthermore, the search can be parallelised for free providing a scalable neural program synthesis approach.
This parallelism and these enumerations algorithms fit the need of scalable search procedures because of the exponential growth of the search space.

![Pipeline for neural predictions for syntax guided program synthesis.\label{fig:description}](main_figure.png)

# How it works?

A user can implement their own DSL. Then following the neural architecures already provided they can create a neural network that predict probabilities for their context free grammar (CFG) generated from their DSL.
We provide an automatic training procedure that supports a few common types off the shelf.
With their network now trained, they can directly use the trained network along with the provided algorithms to find their desired program.
Furthermore, they can scale with multiple CPUs by splitting the work on multiple CPUs with a small overhead at the start and a linear speedup in the number of CPUs. The full technical details of our framework are described in @Fijalkow:2021.

# State of the field

There is mostly one application that enables users to generate programs based on a string description and that is GitHub Copilot [@copilot], it uses neural networks to predict what the user is going to type next. However this project is radically different from ours, first it has been trained on a lot of public code, and generates program without any guarantees based on a textual context whereas we take as input pairs of input-output examples.
Furthermore, many DSL languages with which professionals work with may be limited to internal tools, where our framework can help whereas GitHub Copilot can't. Finally our software is free and open source whereas GitHub Copilot is proprietary software and is indicated to be a paid service.

# Features

The package allows to:
- create a simple DSL from syntaxic constraints and semantics functions;
- Transform this DSL into a context free grammar;
- Transform this CFG into a Probabilistic CFG (PCFG);
- Sample programs from a PCFG;
- Enumerate programs in a PCFG with many different algorithms including `HeapSearch`, a method that is practically faster than $A^*$ while remaining optimal like the latter;
- A grammar splitter that enables to split the search into `n` independent search, enabling easy parallel search without any concerns and scaling linearly with the number of CPUs;
- A neural network architecture to predict probabilities of a CFG given pairs of input-output examples and its automatic training procedure from a DSL that supports `int`, `bool` and `list` inputs.

# References