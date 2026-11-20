function s = extractAfter(str, pat)
  % Octave shim: the part of str after the first occurrence of pat.
  k = strfind(str, pat);
  if isempty(k)   % like MATLAB: no match gives an empty result
    s = "";
    return;
  end
  s = str(k(1) + length(pat):end);
end
