---
title: Growth measurement and models taking into account mechanisms of cell death capture hidden variation in compound response
author:
- name: Song Yi Bae
  affilnum: a
- name: Ning Guan
  affilnum: b
- name: Colton Stearns
  affilnum: b
- name: Aaron S. Meyer
  affilnum: a,c
keywords: [growth models, inference]
affiliation:
- name: Department of Bioengineering, Jonsson Comprehensive Cancer Center, Eli and Edythe Broad Center of Regenerative Medicine and Stem Cell Research; University of California, Los Angeles
  key: a
- name: Massachusetts Institute of Technology, Cambridge, MA
  key: b
- name: Contact info
  key: c
bibliography: ./Manuscript/References.bib
abstract: Cancer cell sensitivity or resistance is almost universally quantified through a direct or surrogate measure of cell number over time. However, compound responses can occur through a number of distinct phenotypic outcomes, including changes in cell growth, apoptosis, and non-apoptotic cell death. These outcomes have distinct effects on the tumor microenvironment, immune response, and capacity for resistance development.
link-citations: true
csl: ./Manuscript/Templates/nature.csl
---

# Summary Points

- Measurements of solely live cell numbers mask important differences in compound effects.
- Automated imaging can provide reasonable throughput to analyze cell response, and endpoint analysis is similarly informative.
- Additive effects on growth and death rates can appear synergistic when analyzed solely via live cell number.
- Synergistic interactions of each individual effect better predict *in vivo* benefit.

# Introduction

Quantifying cellular response to treatment with endogenous or therapeutic compounds is critical to understanding these agents' mechanism of action and therapeutic efficacy. In the case of cancer treatments, and often with other diseases, compounds are evaluated by quantifying the number of cells after a short period of time using direct or surrogate measurements. However, quantities beyond the number and viability of cells provide extremely valuable information about cellular response. Compounds such as targeted kinase inhibitors are known to often operate through cell division rates, as opposed to promoting cell death, allowing for the survival of persister cell populations which enable resistance development [@Hata:2016dp] [@Sharma:2010ge] [@Shaffer:2017a]. Cell death can occur via a variety of mechanisms, including apoptosis and necroptosis, and selection among these outcomes can potently modulate the immunogenicity of a cancer [@Gotwals:2017kj]. Combination treatments are typically evaluated for their ability to enact greater effects than either compound alone, however how this might be evaluated when quantifying multiple cellular phenotypes is unclear.

Here, we show that directly measuring rates of cell death along with cell number over time can provide valuable information for interpreting the response of cells to single and combination compound treatments. Through analysis taking into account the compound-induced changes in rates of cell growth and death, we propose a framework for quantifying sensitivity to compounds and combination effect. This approach reveals extensive differences in cell response otherwise hidden by simply quantifying cell number. Of course, trade-offs exist for the breadth versus depth of analysis that can be performed to characterize cell-compound response. We show that endpoint analysis preserves much of the distinct outcomes we observe, while allowing similarly high-throughput analysis to live cell number surrogates. These results demonstrate the need and an approach to more exactly quantify the nature of cell-compound response and interactions.
