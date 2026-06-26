window.MathJax = {
  loader: {load: ['[tex]/ams', '[tex]/boldsymbol']},
  tex: {
    packages: {'[+]': ['ams', 'boldsymbol']},
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(() => {
  MathJax.typesetPromise()
})
