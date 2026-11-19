function rs = RandStream(varargin)
  % Octave shim: a deterministic stream whose rand(m, n) gives fixed
  % irrational-looking weights, enough for CORA's exponent hashing.
  rs.rand = @(m, n) reshape(mod(sqrt(primes(1e4)(1:m*n)), 1), m, n);
end
