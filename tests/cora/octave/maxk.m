function [b, i] = maxk(a, k)
  % Octave shim: the k largest entries of a vector and their indices.
  [s, j] = sort(a(:).', "descend");
  k = min(k, numel(s));
  b = s(1:k); i = j(1:k);
end
