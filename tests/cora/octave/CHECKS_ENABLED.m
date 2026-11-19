function res = CHECKS_ENABLED()
  % Octave shim: CORA's input validation uses functions Octave lacks
  % (extractAfter and others), so it is switched off for the cross-check.
  res = false;
end
