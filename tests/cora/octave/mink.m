function [b, i] = mink(a, k)
  % Octave shim: the k smallest entries of a vector and their indices.
  [s, j] = sort(a(:).', "ascend");
  k = min(k, numel(s));
  b = s(1:k); i = j(1:k);
end
