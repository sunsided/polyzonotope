classdef Copyable < handle
  % Octave shim: MATLAB's handle class with a shallow copy, the base class
  % of CORA's contDynamics. Octave has no matlab.mixin.Copyable.
  methods
    function c = copy(obj)
      % A new object of the same class with the same property values.
      c = feval(class(obj));
      for p = properties(obj)'
        c.(p{1}) = obj.(p{1});
      end
    end
  end
end
