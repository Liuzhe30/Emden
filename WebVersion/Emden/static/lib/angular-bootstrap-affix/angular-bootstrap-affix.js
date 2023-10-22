'use strict';

/**
 * Firefox document.body.scrollHeight = 0; 
 * Because of that the affix-bottom class is constantly added and removed causing flicker.
 * This function fixes that.
 * Stolen from here: http://james.padolsey.com/snippets/get-document-height-cross-browser/
 */
function getDocHeight() {
  var D = document;
  return Math.max(
      D.body.scrollHeight, D.documentElement.scrollHeight,
      D.body.offsetHeight, D.documentElement.offsetHeight,
      D.body.clientHeight, D.documentElement.clientHeight
  );
}

angular.module('mgcrea.bootstrap.affix', ['mgcrea.jquery']).directive('bsAffix', [
  '$window',
  'dimensions',
  function ($window, dimensions) {
    var checkPosition = function (instance, el, options) {

      var scrollTop = window.pageYOffset;
      var windowHeight = window.innerHeight;
      var scrollHeight = getDocHeight();///document.body.scrollHeight;
      var position = dimensions.offset.call(el[0]);
      var height = dimensions.height.call(el[0]);
      var offsetTop = options.offsetTop * 1;
      var offsetBottom = options.offsetBottom * 1;
      var reset = 'affix affix-top affix-bottom';
      var affix;
      if (instance.originTop == null) {
        instance.originTop = position.top;
      }
      if (windowHeight >= height && instance.originTop <= scrollTop) {
        affix = 'top';
      } else if (windowHeight <= height && scrollTop >= instance.originTop) {
        affix = 'bottom';
      } else {
        affix = false;
      }
      if (instance.affixed === affix)
        return;
      instance.affixed = affix;
      el.removeClass(reset).addClass('' + (affix ? 'affix affix-' + affix : ''));
    };
    var checkCallbacks = function (scope, instance, iElement, iAttrs) {
      if (instance.affixed) {
        if (iAttrs.onUnaffix)
          eval('scope.' + iAttrs.onUnaffix);
      } else {
        if (iAttrs.onAffix)
          eval('scope.' + iAttrs.onAffix);
      }
    };
    return {
      restrict: 'EAC',
      link: function postLink(scope, iElement, iAttrs) {
        var instance = { unpin: null };
        angular.element($window).bind('scroll', function () {
          checkPosition(instance, iElement, iAttrs);
          checkCallbacks(scope, instance, iElement, iAttrs);
        });
        angular.element($window).bind('click', function () {
          setTimeout(function () {
            checkPosition(instance, iElement, iAttrs);
            checkCallbacks(scope, instance, iElement, iAttrs);
          }, 1);
        });
      }
    };
  }
]);